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

# 스크립트 위치 기준 경로 설정
SCRIPT_DIR = Path(__file__).parent
INPUT_DIR = SCRIPT_DIR / 'raw'
OUTPUT_DIR = SCRIPT_DIR / 'complete'

def get_complete_participants():
    """완료 참가자 ID 목록 반환"""

    # 1. 설문 데이터에서 UCLA와 DASS 모두 완료한 참가자 추출
    surveys = pd.read_csv(f'{INPUT_DIR}/2_surveys_results.csv', encoding='utf-8-sig')
    surveys['surveyName'] = surveys['surveyName'].str.lower()

    ucla_participants = set(surveys[surveys['surveyName'] == 'ucla']['participantId'].unique())
    dass_participants = set(surveys[surveys['surveyName'].str.contains('dass', na=False)]['participantId'].unique())

    survey_complete = ucla_participants & dass_participants
    print(f"설문 완료 참가자: {len(survey_complete)}명 (UCLA: {len(ucla_participants)}, DASS: {len(dass_participants)})")

    # 2. 인지과제 데이터에서 PRP, WCST, Stroop 모두 완료한 참가자 추출
    cognitive = pd.read_csv(f'{INPUT_DIR}/3_cognitive_tests_summary.csv', encoding='utf-8-sig')
    cognitive['testName'] = cognitive['testName'].str.lower()

    prp_participants = set(cognitive[cognitive['testName'] == 'prp']['participantId'].unique())
    wcst_participants = set(cognitive[cognitive['testName'] == 'wcst']['participantId'].unique())
    stroop_participants = set(cognitive[cognitive['testName'] == 'stroop']['participantId'].unique())

    cognitive_complete = prp_participants & wcst_participants & stroop_participants
    print(f"인지과제 완료 참가자: {len(cognitive_complete)}명 (PRP: {len(prp_participants)}, WCST: {len(wcst_participants)}, Stroop: {len(stroop_participants)})")

    # 3. 두 조건의 교집합 = 완료 참가자
    complete_participants = survey_complete & cognitive_complete
    print(f"전체 완료 참가자: {len(complete_participants)}명")

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

        # participantId 열로 필터링
        if 'participantId' in df.columns:
            df_filtered = df[df['participantId'].isin(complete_ids)]
        else:
            print(f"  [WARN] {filename}에 participantId 열이 없습니다.")
            df_filtered = df

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
