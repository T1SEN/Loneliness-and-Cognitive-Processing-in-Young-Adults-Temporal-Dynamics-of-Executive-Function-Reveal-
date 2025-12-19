import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
from pathlib import Path

from publication.preprocessing.dataset_builder import build_all_datasets

# 스크립트 위치 기준 경로 설정
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RAW_DIR = SCRIPT_DIR / 'raw'
SERVICE_ACCOUNT_PATH = PROJECT_ROOT / 'serviceAccountKey.json'

RAW_DIR.mkdir(parents=True, exist_ok=True)
if not SERVICE_ACCOUNT_PATH.exists():
    raise FileNotFoundError(f"serviceAccountKey.json 파일을 찾을 수 없습니다: {SERVICE_ACCOUNT_PATH}")

# --- Part 1: Firebase 초기화 ---
try:
    cred = credentials.Certificate(str(SERVICE_ACCOUNT_PATH))
    firebase_admin.initialize_app(cred)
except ValueError:
    print("Firebase 앱이 이미 초기화되었습니다.")

db = firestore.client()

# --- Part 2: 추출한 데이터를 담을 빈 리스트들 준비 ---
# 각 리스트는 별도의 CSV 파일이 됩니다.
participants_data = []          # 1. 참가자 기본 정보
surveys_results_data = []       # 2. 설문 결과 (UCLA, DASS)
cognitive_summary_data = []     # 3. 인지 과제 요약 결과 (PRP, WCST, Stroop)
prp_trial_data = []             # 4. PRP 시행별 데이터
wcst_trial_data = []            # 5. WCST 시행별 데이터
stroop_trial_data = []          # 6. Stroop 시행별 데이터

print("Firebase 데이터 추출을 시작합니다...")

# --- Part 3: 참가자 데이터 가져오기 ---
participants_ref = db.collection('participants').order_by('createdAt', direction=firestore.Query.ASCENDING)
participants_stream = participants_ref.stream()

# --- Part 4: 각 참가자 문서를 순회하며 모든 데이터 처리 ---
for participant_doc in participants_stream:
    participant_id = participant_doc.id
    participant_info = participant_doc.to_dict()

    print(f"\n[참가자 ID: {participant_id}] 처리 시작...")

    # 4-1. 참가자 기본 정보 저장
    participants_data.append({
        'participantId': participant_id,
        'studentId': participant_info.get('studentId'),
        'gender': participant_info.get('gender'),
        'age': participant_info.get('age'),
        'birthDate': participant_info.get('birthDate'),
        'education': participant_info.get('education'),
        'courseName': participant_info.get('courseName'),
        'professorName': participant_info.get('professorName'),
        'classSection': participant_info.get('classSection'),
        'createdAt': participant_info.get('createdAt'),
    })

    # 4-2. Surveys 하위 컬렉션 데이터 추출 (UCLA, DASS)
    surveys_ref = db.collection('participants').document(participant_id).collection('surveys').stream()
    for survey_doc in surveys_ref:
        survey_name = survey_doc.id
        survey_data = survey_doc.to_dict()
        print(f"  - {survey_name} 설문 데이터 처리 중...")

        # 기본 정보
        base_row = {
            'participantId': participant_id,
            'surveyName': survey_name,
            'duration_seconds': survey_data.get('duration_seconds'),
        }

        # survey_data가 비어있는 경우를 대비
        if not survey_data:
            surveys_results_data.append(base_row)
            continue

        # 응답 배열(list)을 q1, q2... 열로 펼치기
        responses = survey_data.get('responses', [])
        if responses:
            for i, resp in enumerate(responses):
                base_row[f'q{i+1}'] = resp
        
        # DASS의 점수 맵(dict)을 score_D, score_A... 열로 펼치기
        if 'scores' in survey_data and isinstance(survey_data['scores'], dict):
            for key, value in survey_data['scores'].items():
                base_row[f'score_{key}'] = value
        
        # UCLA의 단일 점수
        if 'score' in survey_data:
            base_row['score'] = survey_data.get('score')

        surveys_results_data.append(base_row)

    # 4-3. Cognitive Tests 하위 컬렉션 데이터 추출 (PRP, WCST, Stroop)
    tests_ref = db.collection('participants').document(participant_id).collection('cognitive_tests').stream()
    for test_doc in tests_ref:
        test_name = test_doc.id
        test_data = test_doc.to_dict()
        print(f"  - {test_name} 인지과제 데이터 처리 중...")

        # test_data가 비어있는 경우를 대비
        if not test_data:
            continue

        # (A) 요약 데이터 처리
        summary_info = test_data.get('resultsSummary', {})
        summary_row = {
            'participantId': participant_id,
            'testName': test_name,
            'duration_seconds': test_data.get('duration_seconds'),
        }
        # 요약 통계(dict)를 개별 열로 펼치기
        if summary_info:
            for key, value in summary_info.items():
                summary_row[f'{key}'] = value
        
        cognitive_summary_data.append(summary_row)

        # (B) 시행별 상세 데이터 처리
        trials = test_data.get('trialData', [])
        if not trials:
            continue
            
        for trial in trials:
            trial_row = trial.copy()  # 원본 수정을 피하기 위해 복사
            trial_row['participantId'] = participant_id
            
            if test_name == 'prp':
                prp_trial_data.append(trial_row)
            elif test_name == 'wcst':
                wcst_trial_data.append(trial_row)
            elif test_name == 'stroop':
                stroop_trial_data.append(trial_row)


# --- Part 5: 추출한 모든 데이터를 Pandas DataFrame으로 변환 후 CSV 파일로 저장 ---
print("\n모든 데이터 추출 완료! RAW CSV 파일로 저장합니다...")

def save_to_csv(data_list, filename):
    if data_list:
        try:
            df = pd.DataFrame(data_list)
            # participantId를 맨 앞으로 이동
            if 'participantId' in df.columns:
                cols = ['participantId'] + [col for col in df.columns if col != 'participantId']
                df = df[cols]
            
            filepath = RAW_DIR / filename
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"  [SUCCESS] '{filepath}' 파일 저장 완료! (총 {len(df)} 행)")
        except Exception as e:
            print(f"  [ERROR] '{filename}' 파일 저장 실패: {e}")
    else:
        print(f"  [INFO] '{filename}' 파일은 데이터가 없어 생성되지 않았습니다.")

# 각 데이터 리스트를 파일로 저장
save_to_csv(participants_data, '1_participants_info.csv')
save_to_csv(surveys_results_data, '2_surveys_results.csv')
save_to_csv(cognitive_summary_data, '3_cognitive_tests_summary.csv')
save_to_csv(prp_trial_data, '4a_prp_trials.csv')
save_to_csv(wcst_trial_data, '4b_wcst_trials.csv')
save_to_csv(stroop_trial_data, '4c_stroop_trials.csv')

print(f"\n[COMPLETE] 작업 완료! RAW 데이터는 '{RAW_DIR}' 폴더에서 확인할 수 있습니다.")
print("\n[FILTER] RAW 데이터를 기반으로 task별 COMPLETE 폴더를 갱신합니다...")
build_all_datasets(save=True, verbose=True)
