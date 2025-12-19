"""
Task-specific dataset builder.

각 인지검사(Stroop, PRP, WCST)별 complete 데이터셋 빌드.
raw/ 데이터를 필터링하여 complete_{task}/ 디렉토리에 저장.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Set, Optional

import pandas as pd

from .constants import (
    RAW_DIR,
    VALID_TASKS,
    get_results_dir,
)
from .filters import (
    get_task_valid_participants,
    SurveyQCCriteria,
    StroopQCCriteria,
    PRPQCCriteria,
    WCSTQCCriteria,
)

# Windows Unicode 출력 지원
if sys.platform.startswith("win") and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')


TASK_FILES = {
    'stroop': [
        '1_participants_info.csv',
        '2_surveys_results.csv',
        '3_cognitive_tests_summary.csv',
        '4c_stroop_trials.csv',
    ],
    'prp': [
        '1_participants_info.csv',
        '2_surveys_results.csv',
        '3_cognitive_tests_summary.csv',
        '4a_prp_trials.csv',
    ],
    'wcst': [
        '1_participants_info.csv',
        '2_surveys_results.csv',
        '3_cognitive_tests_summary.csv',
        '4b_wcst_trials.csv',
    ],
}


def build_task_dataset(
    task: str,
    data_dir: Path = None,
    output_dir: Path = None,
    survey_criteria: SurveyQCCriteria = None,
    task_criteria = None,
    save: bool = True,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """task별 complete 데이터셋 빌드.

    1. raw/에서 데이터 로드
    2. get_task_valid_participants(task)로 유효 참가자 필터
    3. 6개 CSV 파일 필터링하여 complete_{task}/에 저장

    Args:
        task: 'stroop', 'prp', 'wcst'
        data_dir: raw 데이터 디렉토리 (기본: RAW_DIR)
        output_dir: 출력 디렉토리 (기본: complete_{task}/)
        survey_criteria: 설문 QC 기준
        task_criteria: task별 QC 기준
        save: 파일 저장 여부
        verbose: 상세 로그 출력 여부

    Returns:
        {파일명: DataFrame} 딕셔너리
    """
    if task not in VALID_TASKS:
        raise ValueError(f"Unknown task: {task}. Valid tasks: {VALID_TASKS}")

    if data_dir is None:
        data_dir = RAW_DIR
    if output_dir is None:
        output_dir = get_results_dir(task)

    if verbose:
        print("=" * 60)
        print(f"{task.upper()} 데이터셋 빌드 시작")
        print("=" * 60)

    # 유효 참가자 ID 목록
    valid_ids = get_task_valid_participants(
        task=task,
        data_dir=data_dir,
        survey_criteria=survey_criteria,
        task_criteria=task_criteria,
        verbose=verbose,
    )

    if not valid_ids:
        if verbose:
            print(f"[WARN] {task} 유효 참가자 없음")
        return {}

    if verbose:
        print(f"\n유효 참가자: {len(valid_ids)}명")

    # 출력 폴더 생성
    if save:
        os.makedirs(output_dir, exist_ok=True)

    # 각 파일 필터링
    results: Dict[str, pd.DataFrame] = {}

    for filename in TASK_FILES[task]:
        input_path = data_dir / filename

        if not input_path.exists():
            if verbose:
                print(f"  [SKIP] {filename} 파일 없음")
            continue

        df = pd.read_csv(input_path, encoding='utf-8-sig')
        original_count = len(df)

        # participantId 열 확인
        if 'participantId' not in df.columns:
            if verbose:
                print(f"  [ERROR] {filename}에 participantId 열 없음")
            continue

        # 필터링
        df_filtered = df[df['participantId'].isin(valid_ids)].copy()
        if filename == '3_cognitive_tests_summary.csv' and 'testName' in df_filtered.columns:
            df_filtered['testName'] = df_filtered['testName'].str.lower()
            df_filtered = df_filtered[df_filtered['testName'] == task]
        filtered_count = len(df_filtered)

        results[filename] = df_filtered

        # 저장
        if save:
            output_path = output_dir / filename
            df_filtered.to_csv(output_path, index=False, encoding='utf-8-sig')

        if verbose:
            print(f"  [OK] {filename}: {original_count} -> {filtered_count} 행")

    if verbose:
        print(f"\n완료! '{output_dir}' 폴더에 저장됨")

    # 유효 참가자 ID 목록 저장
    if save:
        ids_path = output_dir / 'filtered_participant_ids.csv'
        pd.DataFrame({'participantId': sorted(valid_ids)}).to_csv(
            ids_path, index=False, encoding='utf-8-sig'
        )
        if verbose:
            print(f"  [OK] 유효 참가자 ID 목록: {ids_path}")

    return results


def build_all_datasets(
    data_dir: Path = None,
    save: bool = True,
    verbose: bool = True,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """stroop, prp, wcst 3개 데이터셋 모두 빌드.

    Args:
        data_dir: raw 데이터 디렉토리 (기본: RAW_DIR)
        save: 파일 저장 여부
        verbose: 상세 로그 출력 여부

    Returns:
        {task: {파일명: DataFrame}} 딕셔너리
    """
    results = {}

    for task in sorted(VALID_TASKS):
        results[task] = build_task_dataset(
            task=task,
            data_dir=data_dir,
            save=save,
            verbose=verbose,
        )
        if verbose:
            print()  # 빈 줄 추가

    return results


def get_dataset_info(task: str = None) -> Dict:
    """데이터셋 정보 조회.

    Args:
        task: 'stroop', 'prp', 'wcst', 또는 None (모든 task)

    Returns:
        데이터셋 정보 딕셔너리
    """
    if task is not None and task not in VALID_TASKS:
        raise ValueError(f"Unknown task: {task}. Valid tasks: {VALID_TASKS}")

    tasks = [task] if task else sorted(VALID_TASKS)
    info = {}

    for t in tasks:
        task_dir = get_results_dir(t)
        participants_path = task_dir / '1_participants_info.csv'

        task_info = {
            'path': str(task_dir),
            'exists': task_dir.exists(),
            'n_participants': 0,
            'files': [],
        }

        if task_dir.exists():
            # 파일 목록
            task_info['files'] = [f.name for f in task_dir.glob('*.csv')]

            # 참가자 수
            if participants_path.exists():
                df = pd.read_csv(participants_path, encoding='utf-8-sig')
                task_info['n_participants'] = len(df)

        info[t] = task_info

    return info


def print_dataset_summary():
    """데이터셋 요약 출력."""
    print("=" * 60)
    print("데이터셋 요약")
    print("=" * 60)

    info = get_dataset_info()

    for task, task_info in info.items():
        status = "OK" if task_info['exists'] else "NO"
        n = task_info['n_participants']
        print(f"  [{status}] {task.upper():8} - N={n:3} ({task_info['path']})")

    # raw 데이터 정보
    raw_participants_path = RAW_DIR / '1_participants_info.csv'
    if raw_participants_path.exists():
        raw_df = pd.read_csv(raw_participants_path, encoding='utf-8-sig')
        print(f"\n  [Raw] N={len(raw_df)} ({RAW_DIR})")


if __name__ == '__main__':
    # 직접 실행 시 모든 데이터셋 빌드
    build_all_datasets()
