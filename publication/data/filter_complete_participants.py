"""
완료 참가자 필터링 스크립트

완료 기준:
- UCLA 설문 완료
- DASS-21 설문 완료
- PRP 인지과제 완료
- WCST 인지과제 완료
- Stroop 인지과제 완료

실행: python -m publication.data.filter_complete_participants
"""

import pandas as pd
import os
import shutil
from pathlib import Path

from publication.preprocessing.constants import (
    DEFAULT_RT_MIN,
    PRP_RT_MAX,
    DEFAULT_SOA_SHORT,
    DEFAULT_SOA_LONG,
)

# 스크립트 위치 기준 경로 설정
SCRIPT_DIR = Path(__file__).parent
INPUT_DIR = SCRIPT_DIR / 'raw'
OUTPUT_DIR = SCRIPT_DIR / 'complete'
RECENT_START_DATE = pd.Timestamp('2025-09-01 00:00:00', tz='UTC')


def get_quality_complete_participants(summary_df):
    """핵심 지표가 정상인 참가자만 반환

    품질 기준:
    - PRP: n_trials == 120, mrt_t2/rt2_soa_50/rt2_soa_1200이 non-null
    - Stroop: mrt_cong/mrt_incong/stroop_effect/accuracy가 non-null
    - WCST: totalTrialCount > 0, completedCategories > 0, perseverativeErrorCount가 non-null
    """
    summary_df = summary_df.copy()
    summary_df['testName'] = summary_df['testName'].str.lower()

    # PRP 품질 기준
    prp_df = summary_df[summary_df['testName'] == 'prp']
    prp_quality_mask = (
        (prp_df['n_trials'] == 120) &
        (prp_df['mrt_t2'].notna()) &
        (prp_df['rt2_soa_50'].notna()) &
        (prp_df['rt2_soa_1200'].notna())
    )
    prp_ok = set(prp_df.loc[prp_quality_mask, 'participantId'])
    prp_all = set(prp_df['participantId'])
    prp_excluded = prp_all - prp_ok

    # Stroop 품질 기준
    stroop_df = summary_df[summary_df['testName'] == 'stroop']
    stroop_quality_mask = (
        (stroop_df['mrt_cong'].notna()) &
        (stroop_df['mrt_incong'].notna()) &
        (stroop_df['stroop_effect'].notna()) &
        (stroop_df['accuracy'].notna())
    )
    stroop_ok = set(stroop_df.loc[stroop_quality_mask, 'participantId'])
    stroop_all = set(stroop_df['participantId'])
    stroop_excluded = stroop_all - stroop_ok

    # WCST 품질 기준
    wcst_df = summary_df[summary_df['testName'] == 'wcst']
    wcst_quality_mask = (
        (wcst_df['totalTrialCount'] > 0) &
        (wcst_df['completedCategories'] > 0) &
        (wcst_df['perseverativeErrorCount'].notna())
    )
    wcst_ok = set(wcst_df.loc[wcst_quality_mask, 'participantId'])
    wcst_all = set(wcst_df['participantId'])
    wcst_excluded = wcst_all - wcst_ok

    # 로그 출력
    if prp_excluded:
        print(f"  [INFO] PRP 품질 기준 미달: {len(prp_excluded)}명")
    if stroop_excluded:
        print(f"  [INFO] Stroop 품질 기준 미달: {len(stroop_excluded)}명")
    if wcst_excluded:
        print(f"  [INFO] WCST 품질 기준 미달: {len(wcst_excluded)}명")

    return prp_ok, stroop_ok, wcst_ok


def get_prp_valid_participants(prp_path: Path) -> tuple[set, set]:
    """
    PRP trial 데이터에서 유효 trial(정답+시간내+RT 범위) 1건 이상 보유한 참가자 집합을 반환.
    반환값: (valid_ids, no_valid_ids)
    """
    prp_trials = pd.read_csv(prp_path, encoding='utf-8-sig')
    prp_trials['t1_correct'] = prp_trials['t1_correct'].fillna(False)
    prp_trials['t2_correct'] = prp_trials['t2_correct'].fillna(False)
    prp_trials['t2_timeout'] = prp_trials.get('t2_timeout', False)
    if isinstance(prp_trials['t2_timeout'], pd.Series):
        prp_trials['t2_timeout'] = prp_trials['t2_timeout'].fillna(False)

    # RT/soa 컬럼 결정
    rt_col = 't2_rt_ms' if 't2_rt_ms' in prp_trials.columns else ('t2_rt' if 't2_rt' in prp_trials.columns else None)
    soa_col = 'soa_nominal_ms' if 'soa_nominal_ms' in prp_trials.columns else ('soa' if 'soa' in prp_trials.columns else None)
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

    # SOA binning (short/long만 사용)
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


def get_complete_participants():
    """완료 참가자 ID 목록 반환"""

    # 1. 설문 데이터에서 UCLA와 DASS 모두 완료한 참가자 추출
    #    (실제 응답이 존재하는 경우만 완료로 인정)
    surveys = pd.read_csv(f'{INPUT_DIR}/2_surveys_results.csv', encoding='utf-8-sig')
    surveys['surveyName'] = surveys['surveyName'].str.lower()

    # UCLA: score가 non-null인 경우만 완료
    ucla_df = surveys[surveys['surveyName'] == 'ucla']
    ucla_all = set(ucla_df['participantId'].unique())
    ucla_participants = set(ucla_df[ucla_df['score'].notna()]['participantId'].unique())
    ucla_missing_score = ucla_all - ucla_participants
    if ucla_missing_score:
        print(f"  [INFO] UCLA score 누락으로 제외: {len(ucla_missing_score)}명")

    # DASS: score_D, score_A, score_S 모두 non-null인 경우만 완료
    dass_df = surveys[surveys['surveyName'].str.contains('dass', na=False)]
    dass_all = set(dass_df['participantId'].unique())
    dass_participants = set(dass_df[
        (dass_df['score_D'].notna()) &
        (dass_df['score_A'].notna()) &
        (dass_df['score_S'].notna())
    ]['participantId'].unique())
    dass_missing_score = dass_all - dass_participants
    if dass_missing_score:
        print(f"  [INFO] DASS 점수 누락으로 제외: {len(dass_missing_score)}명")

    survey_complete = ucla_participants & dass_participants
    print(f"설문 완료 참가자: {len(survey_complete)}명 (UCLA: {len(ucla_participants)}/{len(ucla_all)}, DASS: {len(dass_participants)}/{len(dass_all)})")

    # 2. 인지과제 데이터에서 PRP, WCST, Stroop 모두 완료하고 품질 기준 충족한 참가자 추출
    cognitive = pd.read_csv(f'{INPUT_DIR}/3_cognitive_tests_summary.csv', encoding='utf-8-sig')

    # 기본 존재 여부 (단순 참가자 수)
    cognitive_lower = cognitive.copy()
    cognitive_lower['testName'] = cognitive_lower['testName'].str.lower()
    prp_all = set(cognitive_lower[cognitive_lower['testName'] == 'prp']['participantId'].unique())
    wcst_all = set(cognitive_lower[cognitive_lower['testName'] == 'wcst']['participantId'].unique())
    stroop_all = set(cognitive_lower[cognitive_lower['testName'] == 'stroop']['participantId'].unique())
    print(f"인지과제 수행 참가자: PRP {len(prp_all)}명, WCST {len(wcst_all)}명, Stroop {len(stroop_all)}명")

    # 품질 기준 적용
    print("품질 기준 적용 중...")
    prp_ok, stroop_ok, wcst_ok = get_quality_complete_participants(cognitive)

    cognitive_complete = prp_ok & stroop_ok & wcst_ok
    print(f"인지과제 품질 기준 통과 참가자: {len(cognitive_complete)}명 (PRP: {len(prp_ok)}, Stroop: {len(stroop_ok)}, WCST: {len(wcst_ok)})")

    # 3. 두 조건의 교집합 = 완료 참가자
    complete_participants = survey_complete & cognitive_complete
    print(f"전체 완료 참가자: {len(complete_participants)}명")

    # 3-1. 성별 누락 참가자 제외
    participants_info_path = INPUT_DIR / '1_participants_info.csv'
    if participants_info_path.exists():
        participants_df = pd.read_csv(participants_info_path, encoding='utf-8-sig')
        missing_gender_ids = set(
            participants_df[
                participants_df['gender'].isna() |
                (participants_df['gender'].astype(str).str.strip() == '')
            ]['participantId'].dropna()
        )
        if missing_gender_ids:
            to_remove = complete_participants & missing_gender_ids
            if to_remove:
                print(f"  [INFO] 성별 누락으로 제외: {len(to_remove)}명")
                complete_participants -= to_remove
    else:
        print("  [WARN] participants_info.csv를 찾을 수 없어 성별 누락 여부를 확인하지 못했습니다.")

    # 3-2. PRP 유효 trial 0건 참가자 제외
    prp_path = INPUT_DIR / '4a_prp_trials.csv'
    if prp_path.exists():
        _, prp_no_valid = get_prp_valid_participants(prp_path)
        to_remove = complete_participants & prp_no_valid
        if to_remove:
            print(f"  [INFO] PRP 유효 trial 0건으로 제외: {len(to_remove)}명")
            complete_participants -= to_remove
    else:
        print("  [WARN] 4a_prp_trials.csv를 찾을 수 없어 PRP 유효 trial 검증을 건너뜁니다.")

    # 4. createdAt 기준 날짜 필터링 (2025-09-01 이후 데이터만)
    if participants_info_path.exists():
        participants_df = pd.read_csv(participants_info_path, encoding='utf-8-sig')
        if 'createdAt' in participants_df.columns:
            participants_df['createdAt'] = pd.to_datetime(participants_df['createdAt'], errors='coerce', utc=True)

            # createdAt 누락 참가자 감지 및 로그
            all_participant_ids = set(participants_df['participantId'].dropna())
            missing_created = set(
                participants_df[participants_df['createdAt'].isna()]['participantId'].dropna()
            )
            # 완료 참가자 중 createdAt 누락된 사람
            missing_in_complete = missing_created & complete_participants
            if missing_in_complete:
                print(f"  [WARN] createdAt 누락으로 제외: {len(missing_in_complete)}명")
                sample_ids = list(missing_in_complete)[:5]
                print(f"    제외된 ID 샘플: {sample_ids}")

            # 날짜 필터 적용 (누락자 제외)
            recent_ids = set(
                participants_df[participants_df['createdAt'] >= RECENT_START_DATE]['participantId'].dropna()
            )
            print(f"createdAt 필터 적용(>= {RECENT_START_DATE.date()}): {len(recent_ids)}명")

            # 날짜 기준 미달로 제외된 참가자 로그
            date_excluded = complete_participants - recent_ids - missing_in_complete
            if date_excluded:
                print(f"  [INFO] 날짜 기준 미달로 제외: {len(date_excluded)}명")
                sample_ids = list(date_excluded)[:5]
                print(f"    제외된 ID 샘플: {sample_ids}")

            complete_participants = complete_participants & recent_ids
            print(f"날짜 필터 반영 후 complete 참가자 {len(complete_participants)}명")
        else:
            print("  [WARN] participants_info.csv에 createdAt 열이 없어 날짜 필터를 적용하지 못했습니다.")
    else:
        print("  [WARN] participants_info.csv 파일을 찾을 수 없어 날짜 필터를 적용하지 못했습니다.")

    return complete_participants


def filter_and_save():
    """모든 CSV 파일을 필터링하여 complete_only 폴더에 저장"""

    complete_ids = get_complete_participants()

    # 출력 폴더 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 필터링할 파일 목록
    files = [
        '1_participants_info.csv',
        '2_surveys_results.csv',
        '3_cognitive_tests_summary.csv',
        '4a_prp_trials.csv',
        '4b_wcst_trials.csv',
        '4c_stroop_trials.csv',
    ]

    for filename in files:
        input_path = f'{INPUT_DIR}/{filename}'
        output_path = f'{OUTPUT_DIR}/{filename}'

        if not os.path.exists(input_path):
            print(f"  [SKIP] {filename} 파일이 없습니다.")
            continue

        df = pd.read_csv(input_path, encoding='utf-8-sig')
        original_count = len(df)

        # participantId 열로 필터링 (없으면 즉시 실패)
        if 'participantId' not in df.columns:
            raise KeyError(
                f"{filename} 파일에 participantId 열이 없습니다. "
                "완료 참가자 필터링이 불가능하므로 스크립트를 종료합니다."
            )
        df_filtered = df[df['participantId'].isin(complete_ids)]

        filtered_count = len(df_filtered)

        # 저장
        df_filtered.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"  [OK] {filename}: {original_count} -> {filtered_count} 행")

    print(f"\n완료! '{OUTPUT_DIR}' 폴더에 필터링된 CSV 파일들이 저장되었습니다.")


if __name__ == '__main__':
    print("=" * 60)
    print("완료 참가자 필터링 시작")
    print("=" * 60)
    filter_and_save()
