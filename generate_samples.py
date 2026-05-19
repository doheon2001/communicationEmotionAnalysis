import os
import json
import subprocess
import sys
import pandas as pd

# Define paths
workspace_dir = os.path.dirname(os.path.abspath(__file__))
raw_dir = os.path.join(workspace_dir, "data", "raw")
os.makedirs(raw_dir, exist_ok=True)

aihub_xlsx_path = os.path.join(raw_dir, "018.감성대화", "Training_221115_add", "원천데이터", "감성대화말뭉치(최종데이터)_Training.xlsx")
aihub_json_path = os.path.join(raw_dir, "018.감성대화", "Training_221115_add", "라벨링데이터", "감성대화말뭉치(최종데이터)_Training.json")
k_haters_merged_path = os.path.join(raw_dir, "Selectstar_Tunip_HUMANELab_opendata", "Selectstar_Tunip_HUMANELab_opendata_merged.json")

# Output files
aihub_out = os.path.join(raw_dir, "aihub_emotion_sample.csv")
k_haters_out = os.path.join(raw_dir, "k_haters_sample.csv")
k_mhas_out = os.path.join(raw_dir, "k_mhas_sample.csv")
team_chat_out = os.path.join(raw_dir, "team_chat_dummy.csv")

# Ensure required libraries are installed
def install_dependencies():
    required = {"pandas", "openpyxl", "datasets"}
    installed = {pkg.split("==")[0].lower() for pkg in subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode().split()}
    missing = required - installed
    if missing:
        print(f"Installing missing libraries: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
    else:
        print("All required libraries (pandas, openpyxl, datasets) are already installed.")

def sample_aihub_emotion():
    print("\n--- 1. Generating AI Hub Emotion Sample ---")
    
    # Try reading the XLSX file first
    if os.path.exists(aihub_xlsx_path):
        print(f"Reading local XLSX file: {aihub_xlsx_path}")
        try:
            df = pd.read_excel(aihub_xlsx_path)
            # Filter rows with non-empty '사람문장1' and '감정_대분류'
            df = df.dropna(subset=['사람문장1', '감정_대분류'])
            
            target_classes = ['기쁨', '슬픔', '분노']
            df_filtered = df[df['감정_대분류'].isin(target_classes)]
            
            samples = []
            for cls in target_classes:
                df_cls = df_filtered[df_filtered['감정_대분류'] == cls]
                # Sample 100 or all if less
                sampled_cls = df_cls.sample(n=min(100, len(df_cls)), random_state=42)
                samples.append(sampled_cls)
            
            df_final = pd.concat(samples)
            df_final = df_final.rename(columns={'사람문장1': 'sentence', '감정_대분류': 'label'})
            df_final = df_final[['sentence', 'label']]
            
            df_final.to_csv(aihub_out, index=False, encoding='utf-8-sig')
            print(f"Saved AI Hub Emotion sample to {aihub_out} ({len(df_final)} rows)")
            return
        except Exception as e:
            print(f"Error reading Excel file: {e}. Falling back to JSON...")
            
    # JSON Fallback
    if os.path.exists(aihub_json_path):
        print(f"Reading local JSON file: {aihub_json_path}")
        try:
            with open(aihub_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Map E-codes to classes
            # Normally E10-E20 contains 분노/슬픔/기쁨 codes
            # E18 is '한심한' (슬픔), E20 is '억울한' (분노), E54 is '신나는' (기쁨) etc.
            # Let's map a few common ones we know or extract based on basic logic
            # Since JSON has no direct text mapping, let's categorize them by emotion-id prefix/suffix or situation:
            # Let's read the JSON structures and map the type code.
            # To be robust, let's parse a mapping based on E-codes:
            # E10~E19: 슬픔/상처, E20~E29: 분노, E50~E59: 기쁨
            # If we inspect the JSON, we can do a smart classification:
            # Let's assign:
            # - '기쁨': E50, E51, E52, E53, E54, E55, E56, E57, E58, E59
            # - '슬픔': E10, E11, E12, E13, E14, E15, E16, E17, E18, E19, E30, E31, E32, E33, E34, E35, E36, E37, E38, E39
            # - '분노': E20, E21, E22, E23, E24, E25, E26, E27, E28, E29, E40, E41, E42, E43, E44, E45, E46, E47, E48, E49
            
            rows = []
            for item in data:
                profile = item.get('profile', {})
                emotion = profile.get('emotion', {})
                etype = emotion.get('type', '')
                talk = item.get('talk', {})
                content = talk.get('content', {})
                sentence = content.get('HS01', '')
                
                if not sentence:
                    continue
                
                label = None
                try:
                    num = int(etype[1:])
                    if 50 <= num <= 59:
                        label = '기쁨'
                    elif (10 <= num <= 19) or (30 <= num <= 39):
                        label = '슬픔'
                    elif (20 <= num <= 29) or (40 <= num <= 49):
                        label = '분노'
                except:
                    continue
                
                if label:
                    rows.append({'sentence': sentence, 'label': label})
            
            df = pd.DataFrame(rows)
            target_classes = ['기쁨', '슬픔', '분노']
            samples = []
            for cls in target_classes:
                df_cls = df[df['label'] == cls]
                sampled_cls = df_cls.sample(n=min(100, len(df_cls)), random_state=42)
                samples.append(sampled_cls)
            
            df_final = pd.concat(samples)
            df_final.to_csv(aihub_out, index=False, encoding='utf-8-sig')
            print(f"Saved AI Hub Emotion sample (JSON fallback) to {aihub_out} ({len(df_final)} rows)")
        except Exception as e:
            print(f"Failed to generate from JSON: {e}")
    else:
        print("No AI Hub Emotion raw dataset found.")

def sample_k_haters():
    print("\n--- 2. Generating K-HATERS Sample ---")
    
    # Try loading from Hugging Face first
    try:
        print("Attempting to load humane-lab/K-HATERS from Hugging Face...")
        from datasets import load_dataset
        dataset = load_dataset("humane-lab/K-HATERS", split="train")
        
        # Check columns
        df = pd.DataFrame(dataset)
        text_col = 'text' if 'text' in df.columns else ('comment' if 'comment' in df.columns else '문장')
        label_col = 'label'
        
        # Get label mapping if label is represented as ClassLabel integers
        if df[label_col].dtype in ['int64', 'int32']:
            try:
                label_names = dataset.features[label_col].names
                df['label_str'] = df[label_col].map(lambda x: label_names[x])
                label_col = 'label_str'
            except Exception as ex:
                print(f"Could not map integer labels: {ex}")
        
        target_classes = ['Normal', 'Offensive', 'L1_hate', 'L2_hate']
        samples = []
        for cls in target_classes:
            df_cls = df[df[label_col] == cls]
            sampled_cls = df_cls.sample(n=min(100, len(df_cls)), random_state=42)
            samples.append(sampled_cls)
            
        df_final = pd.concat(samples)
        df_final = df_final.rename(columns={text_col: 'sentence', label_col: 'label'})
        df_final = df_final[['sentence', 'label']]
        
        df_final.to_csv(k_haters_out, index=False, encoding='utf-8-sig')
        print(f"Saved K-HATERS sample from Hugging Face to {k_haters_out} ({len(df_final)} rows)")
        return
    except Exception as e:
        print(f"Hugging Face load failed ({e}). Falling back to local merged JSON...")

    # Fallback to local merged JSON
    if os.path.exists(k_haters_merged_path):
        print(f"Reading local merged JSON: {k_haters_merged_path}")
        try:
            with open(k_haters_merged_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            rows = []
            for item in data:
                sentence = item.get('문장', '')
                if not sentence:
                    continue
                
                # Hierarchical label classification
                p1 = item.get('1단계Y/N', '')
                p2_1 = item.get('2-1단계Y/N', '')
                p2_2 = item.get('2-2단계Y/N', '')
                
                if p1 == 'N':
                    label = 'Normal'
                elif p1 == 'Y':
                    if p2_1 == 'N':
                        label = 'Offensive'
                    elif p2_1 == 'Y':
                        if p2_2 == 'N':
                            label = 'L1_hate'
                        elif p2_2 == 'Y':
                            label = 'L2_hate'
                        else:
                            continue
                    else:
                        continue
                else:
                    continue
                
                rows.append({'sentence': sentence, 'label': label})
                
            df = pd.DataFrame(rows)
            target_classes = ['Normal', 'Offensive', 'L1_hate', 'L2_hate']
            samples = []
            for cls in target_classes:
                df_cls = df[df['label'] == cls]
                sampled_cls = df_cls.sample(n=min(100, len(df_cls)), random_state=42)
                samples.append(sampled_cls)
                
            df_final = pd.concat(samples)
            df_final.to_csv(k_haters_out, index=False, encoding='utf-8-sig')
            print(f"Saved K-HATERS sample (local fallback) to {k_haters_out} ({len(df_final)} rows)")
        except Exception as e:
            print(f"Failed to generate K-HATERS from local fallback: {e}")
    else:
        print("No local K-HATERS raw dataset found.")

def sample_k_mhas():
    print("\n--- 3. Generating K-MHAS Sample ---")
    try:
        from datasets import load_dataset
        print("Loading jeanlee/kmhas_korean_hate_speech from Hugging Face...")
        dataset = load_dataset("jeanlee/kmhas_korean_hate_speech", split="train")
        
        df = pd.DataFrame(dataset)
        # Columns in K-MHAS: 'text' (comment), 'label' (list of label IDs)
        
        # Classes:
        # 0: Origin (출신차별)
        # 1: Physical (외모차별)
        # 2: Politics (정치성향차별)
        # 3: Profanity (혐오욕설)
        class_mapping = {
            0: 'Origin',
            1: 'Physical',
            2: 'Politics',
            3: 'Profanity'
        }
        
        samples = []
        for class_idx, class_name in class_mapping.items():
            # Select samples that contain exactly this single label
            df_cls = df[df['label'].map(lambda x: len(x) == 1 and x[0] == class_idx)]
            sampled_cls = df_cls.sample(n=min(100, len(df_cls)), random_state=42).copy()
            sampled_cls['label_str'] = class_name
            samples.append(sampled_cls)
            
        df_final = pd.concat(samples)
        df_final = df_final.rename(columns={'text': 'sentence', 'label_str': 'label'})
        df_final = df_final[['sentence', 'label']]
        
        df_final.to_csv(k_mhas_out, index=False, encoding='utf-8-sig')
        print(f"Saved K-MHAS sample to {k_mhas_out} ({len(df_final)} rows)")
    except Exception as e:
        print(f"Failed to generate K-MHAS sample: {e}")

def generate_team_chat_dummy():
    print("\n--- 4. Generating Team Chat Dummy Logs ---")
    
    # Pre-defined 30 high-quality chat entries
    dummy_data = [
        {"speaker": "민우 (PM)", "message": "여러분, 오늘 예정된 마일스톤 배포 준비는 잘 진행되고 있나요?", "timestamp": "14:00:00", "emotion": "기쁨"},
        {"speaker": "서연 (FE)", "message": "어... 방금 배포 테스트 서버에서 메인 대시보드 렌더링이 깨지는 현상이 발견되었어요. 조금 당황스럽네요.", "timestamp": "14:02:00", "emotion": "당황"},
        {"speaker": "준호 (BE)", "message": "어제 API 스키마 변경 사항 반영하셨나요? 제가 백엔드 코드를 보냈는데 피드백이 없으셨던 것 같아요.", "timestamp": "14:03:00", "emotion": "불안"},
        {"speaker": "서연 (FE)", "message": "아, 제가 확인을 늦게 했네요! 스키마가 바뀐 줄 몰랐어요. 배포 1시간 전인데 큰일이네요.", "timestamp": "14:04:00", "emotion": "당황"},
        {"speaker": "민우 (PM)", "message": "배포 일정이 급한데 이런 일이 생기다니 조금 당황스럽지만, 지금 당장 해결 가능한 부분인가요?", "timestamp": "14:05:00", "emotion": "불안"},
        {"speaker": "준호 (BE)", "message": "바뀐 부분은 몇 가지 필드명 뿐이라서 프론트엔드에서 컴포넌트 데이터 매핑만 수정해주시면 될 겁니다.", "timestamp": "14:07:00", "emotion": "안도"},
        {"speaker": "서연 (FE)", "message": "다행이네요. 바로 수정해서 빌드 돌려볼게요. 15분 정도 걸릴 것 같습니다!", "timestamp": "14:08:00", "emotion": "기쁨"},
        {"speaker": "지은 (UI/UX)", "message": "아! 그리고 디자인 검수 진행 중인데, 모바일 화면에서 버튼 텍스트가 잘리는 것 같아요.", "timestamp": "14:10:00", "emotion": "당황"},
        {"speaker": "민우 (PM)", "message": "모바일 뷰에서도 디자인 규격에 맞게 나오도록 수정이 가능한가요?", "timestamp": "14:11:00", "emotion": "불안"},
        {"speaker": "지은 (UI/UX)", "message": "여백 설정(padding)만 아주 살짝 줄이면 될 것 같아요. 피그마 링크 전달드립니다!", "timestamp": "14:12:00", "emotion": "안도"},
        {"speaker": "서연 (FE)", "message": "지은님, 피그마 확인했어요! 대시보드 스키마 수정하는 김에 모바일 여백도 같이 패치해서 올릴게요.", "timestamp": "14:15:00", "emotion": "기쁨"},
        {"speaker": "준호 (BE)", "message": "서연님, 혹시 백엔드 로그에 에러 메시지가 찍히면 저한테 바로 말씀해주세요. 대기하고 있겠습니다.", "timestamp": "14:16:00", "emotion": "안도"},
        {"speaker": "서연 (FE)", "message": "네, 감사해요 준호님! 빌드 시작했습니다.", "timestamp": "14:18:00", "emotion": "안도"},
        {"speaker": "민우 (PM)", "message": "다들 빠르게 협조해주셔서 다행입니다. 이번 배포가 정말 중요해서 온 신경이 곤두서 있네요.", "timestamp": "14:20:00", "emotion": "불안"},
        {"speaker": "지은 (UI/UX)", "message": "사용성 테스트 피드백이 너무 잘 나와서, 이번 업데이트 배포만 무사히 끝나면 유저들이 아주 기뻐할 것 같아요!", "timestamp": "14:22:00", "emotion": "기쁨"},
        {"speaker": "서연 (FE)", "message": "빌드가 성공했어요! 로컬 테스트에서는 렌더링 정상적으로 잘 되는 것 확인했습니다.", "timestamp": "14:25:00", "emotion": "기쁨"},
        {"speaker": "준호 (BE)", "message": "와! 고생하셨습니다. 저도 스테이징 서버 로그 확인해볼게요.", "timestamp": "14:26:00", "emotion": "기쁨"},
        {"speaker": "민우 (PM)", "message": "정말 다행이네요. 바로 스테이징 환경 배포 요청하겠습니다.", "timestamp": "14:27:00", "emotion": "안도"},
        {"speaker": "서연 (FE)", "message": "어... 그런데 왜 스테이징 서버에 반영이 안 될까요? 캐시 문제인가요?", "timestamp": "14:32:00", "emotion": "당황"},
        {"speaker": "준호 (BE)", "message": "아, 제가 스테이징 DB 마이그레이션 스크립트를 실행 안 했었네요. 죄송합니다! 금방 할게요.", "timestamp": "14:33:00", "emotion": "당황"},
        {"speaker": "민우 (PM)", "message": "아... 준호님, 배포 전 체크리스트에 마이그레이션이 포함되어 있었는데 빠뜨리시면 곤란합니다.", "timestamp": "14:35:00", "emotion": "분노"},
        {"speaker": "준호 (BE)", "message": "정말 죄송합니다. 마음이 급하다 보니 실수가 있었네요. 지금 바로 마이그레이션 완료했습니다!", "timestamp": "14:36:00", "emotion": "슬픔"},
        {"speaker": "서연 (FE)", "message": "이제 정상적으로 로드되는 것 확인했습니다! 버튼 여백도 모바일에서 예쁘게 잘 나오네요.", "timestamp": "14:38:00", "emotion": "기쁨"},
        {"speaker": "지은 (UI/UX)", "message": "우와, 정말 마음에 들어요! 서연님 고생 많으셨습니다.", "timestamp": "14:40:00", "emotion": "기쁨"},
        {"speaker": "민우 (PM)", "message": "좋습니다. 검수가 끝났으니 프로덕션 배포 진행하겠습니다.", "timestamp": "14:45:00", "emotion": "안도"},
        {"speaker": "준호 (BE)", "message": "프로덕션 서버 모니터링 시작했습니다. CPU랑 메모리 다 안정적이네요.", "timestamp": "14:47:00", "emotion": "안도"},
        {"speaker": "서연 (FE)", "message": "네, 프로덕션 페이지 접속도 원활하고 기능 동작도 아주 매끄러워요!", "timestamp": "14:50:00", "emotion": "기쁨"},
        {"speaker": "지은 (UI/UX)", "message": "고생 많으셨습니다, 모두들! 오늘 밤은 두 다리 뻗고 잘 수 있겠어요.", "timestamp": "14:52:00", "emotion": "기쁨"},
        {"speaker": "민우 (PM)", "message": "다들 너무 고생하셨습니다. 내일 점심은 맛있는 거 먹으러 가요!", "timestamp": "14:55:00", "emotion": "기쁨"},
        {"speaker": "준호 (BE)", "message": "좋습니다! 다들 푹 쉬시고 내일 뵙겠습니다!", "timestamp": "15:00:00", "emotion": "기쁨"}
    ]
    
    df = pd.DataFrame(dummy_data)
    df.to_csv(team_chat_out, index=False, encoding='utf-8-sig')
    print(f"Saved Team Chat dummy logs to {team_chat_out} ({len(df)} rows)")

def main():
    install_dependencies()
    sample_aihub_emotion()
    sample_k_haters()
    # sample_k_mhas()  # Commented out as K-MHAS source files are not present locally yet
    generate_team_chat_dummy()
    print("\nAll datasets generated successfully!")

if __name__ == "__main__":
    main()
