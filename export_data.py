import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd

# --- Part 1: Firebase 초기화 ---
try:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
except ValueError:
    print("Firebase 앱이 이미 초기화되었습니다.")

db = firestore.client()

# --- Part 2: 데이터를 담을 빈 리스트 준비 ---
participants_data = []

print("참가자 기본 정보 추출을 시작합니다...")

# --- Part 3: 'participants' 컬렉션에서 'createdAt' 기준으로 정렬하여 가져오기 ---
# direction=firestore.Query.ASCENDING 은 '오름차순', 즉 실험 먼저 마친 순서입니다.
participants_ref = db.collection('participants').order_by('createdAt', direction=firestore.Query.ASCENDING)
participants_stream = participants_ref.stream()

# --- Part 4: 각 참가자 문서를 순회하며 데이터 처리 ---
for participant_doc in participants_stream:
    participant_id = participant_doc.id
    participant_info = participant_doc.to_dict()

    # CSV 한 행에 들어갈 데이터 구성
    # .get() 메소드를 사용하면 혹시 필드가 없더라도 오류 없이 None으로 처리됩니다.
    single_participant_data = {
        'participantId': participant_id,
        'age': participant_info.get('age'),
        'birthDate': participant_info.get('birthDate'),
        'classSection': participant_info.get('classSection'),
        'courseName': participant_info.get('courseName'),
        'createdAt': participant_info.get('createdAt'),
        'education': participant_info.get('education'),
        'gender': participant_info.get('gender'),
        'professorName': participant_info.get('professorName'),
        'studentId': participant_info.get('studentId'),
    }
    
    participants_data.append(single_participant_data)
    print(f"참가자 ID: {participant_id} 정보 추출 완료.")

print(f"\n총 {len(participants_data)}명의 참가자 데이터 추출 완료!")
print("CSV 파일로 저장합니다...")

# --- Part 5: 리스트를 Pandas DataFrame으로 변환 후 CSV 파일로 저장 ---
if participants_data:
    # 컬럼 순서를 보기 좋게 정렬합니다.
    ordered_columns = [
        'participantId', 'studentId', 'gender', 'age', 'birthDate', 
        'education', 'courseName', 'professorName', 'classSection', 'createdAt'
    ]
    
    df_participants = pd.DataFrame(participants_data)
    
    # 실제 데이터에 있는 컬럼만 남기고 순서를 맞춥니다.
    final_columns = [col for col in ordered_columns if col in df_participants.columns]
    df_participants = df_participants[final_columns]
    
    # encoding='utf-8-sig'는 엑셀에서 한글이 깨지지 않게 합니다.
    df_participants.to_csv("participants_info.csv", index=False, encoding='utf-8-sig')

    print("\n작업 완료! 폴더에서 'participants_info.csv' 파일을 확인하세요.")
else:
    print("추출할 참가자 데이터가 없습니다.")