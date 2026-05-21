# data/raw 샘플 데이터

`data/raw/`의 `*_raw_sample.csv` 파일은 AIHub·K-HATERS·K-MHaS 원본에서 **일부만 추출**한 것입니다.  
**원본과 동일한 행·열 구조**를 유지하며, 샘플링·인코딩·부가 설명만 이 문서에 적습니다.

## 공통

| 항목 | 내용 |
| --- | --- |
| 인코딩 | UTF-8 (BOM: `utf-8-sig`로 CSV 저장) |
| 샘플링 | `random_state=42`로 무작위 추출, 건수는 파일명에 표기 |
| 재생성 | 저장소 루트에서 `python generate_raw_samples.py` |

원본 전체는 용량 때문에 git에 포함하지 않습니다. UTF-8 사본은 `018.감성대화/`, `talksets/`, `Selectstar_Tunip_HUMANELab_opendata/`, `K-MHaS/` 등에 둡니다.

---

## aihub_emotion_raw_sample.csv (300건)

| 항목 | 내용 |
| --- | --- |
| 원본 | `감성대화말뭉치(최종데이터)_Training.xlsx` |
| 행 | 원본 1행 = 대화 1건 (동일) |
| 열 (13개) | `Unnamed: 0`, `연령`, `성별`, `상황키워드`, `전체선톡`, `감정_대분류`, `감정_소분류`, `사람문장1`, `시스템문장1`, `사람문장2`, `시스템문장2`, `사람문장3`, `시스템문장3` |

`Unnamed: 0`은 XLSX를 pandas로 읽을 때 생기는 인덱스 열 이름으로, 원본 표와 동일합니다.

---

## k_haters_raw_sample.csv (400건)

| 항목 | 내용 |
| --- | --- |
| 원본 | `Selectstar_Tunip_HUMANELab_opendata_merged.json` (개별 JSON 병합본) |
| 행 | 원본 1레코드 = 댓글 1건 (동일) |
| 열 (27개) | `file_name`, `모욕`, `욕설`, `외설`, `폭력위협/범죄조장`, `성혐오`, `연령`, `인종/지역`, `장애`, `종교`, `정치성향`, `직업`, `1단계Y/N`, `2-1단계Y/N`, `2-2단계Y/N`, `문장`, `대상하이라이트`, `정치성향Y/N`, `혐오 클래스`, `특정 집단`, `그 외`, `인종/지역Y/N`, `성혐오Y/N`, `연령Y/N`, `장애Y/N`, `종교Y/N`, `직업Y/N` |

라벨 해석(예: `1단계Y/N` → Normal / Offensive / L1_hate / L2_hate)은 `data/processed/` 단계 또는 `generate_samples.py`에서 수행합니다.

---

## aihub_talkset_raw_sample.csv (200건)

| 항목 | 내용 |
| --- | --- |
| 원본 | `talksets-train-1.txt` ~ `talksets-train-5.txt` |
| 행 | 원본 **1줄 = 다자간 대화 1건** (동일) |
| 열 | 원본 txt와 같이 **컬럼 헤더 없음**, 한 줄에 대화 전체 1필드 |

### 줄 형식 (원본 txt와 동일)

```
{발화1|발화2|발화3|...}
```

- 중괄호 `{}` 안에 발화를 `|`로 구분합니다.
- CSV는 필드에 쉼표·따옴표가 있을 수 있어 RFC 4180 따옴표 처리만 적용했습니다. **추가 컬럼은 없습니다.**

### 라벨 JSON (참고용, 샘플 CSV에 미포함)

같은 줄 순서에 대응하는 라벨 파일:

| txt | JSON |
| --- | --- |
| talksets-train-1.txt | talksets-train-1_aihub.json |
| talksets-train-2.txt | talksets-train-2.json |
| talksets-train-3.txt | talksets-train-3.json |
| talksets-train-4.txt | talksets-train-4.json |
| talksets-train-5.txt | talksets-train-5.json |

JSON의 `sentences[]` 필드 예시 (문장 단위 라벨, **원본 txt에는 없음**):

| 필드 | 의미 |
| --- | --- |
| `id` | 문장 ID |
| `speaker` | 화자 번호 (1, 2, …) |
| `origin_text` / `text` | 원문 / 정제문 |
| `types` | 윤리 유형 (`CENSURE`, `HATE`, `SEXUAL`, `IMMORAL_NONE` 등) |
| `is_immoral` | 비윤리 여부 |
| `intensity` | 불쾌·공격 강도 (비윤리면 보통 0) |
| `intensity_sum` | 라벨러 투표 합 |
| `votes` | 라벨러별 강도·인구통계 (비윤리 문장은 `null`) |
| `frame_id` | 의미 프레임 ID (없으면 0) |
| `mapped_slots` | 형태·의미 슬롯 (없으면 `[]`) |

한 대화 안에서도 발화마다 `IMMORAL_NONE`(비윤리)과 `SEXUAL` 등이 섞일 수 있습니다. 문장 단위 분석이 필요하면 위 JSON을 사용하세요.

---

## k_mhas_raw_sample.csv (400건)

| 항목 | 내용 |
| --- | --- |
| 원본 | [K-MHaS](https://github.com/adlnlp/K-MHaS) `data/kmhas_train.txt` (학습 78,977건) |
| 행 | 원본 1행 = 뉴스 댓글 1건 (동일) |
| 열 (2개) | `document`, `label` |

### 원본 형식

- 원본 txt는 **탭(`\t`) 구분**이며, 샘플 CSV는 동일한 컬럼명으로 저장했습니다.
- `label`은 **쉼표로 구분된 정수 ID 문자열**입니다 (멀티레이블). 예: `2,4`, `8`, `2,3,4`
- 영문 클래스명(`Origin`, `Politics` 등)으로 바꾸지 않았습니다. ID → 이름 매핑은 `data/processed/` 단계에서 수행합니다.

### 라벨 ID (0~8)

| ID | 영문 | 한글 |
| --- | --- | --- |
| 0 | Origin | 출신차별 |
| 1 | Physical | 외모차별 |
| 2 | Politics | 정치성향차별 |
| 3 | Profanity | 혐오욕설 |
| 4 | Age | 연령차별 |
| 5 | Gender | 성차별 |
| 6 | Race | 인종차별 |
| 7 | Religion | 종교차별 |
| 8 | Not Hate Speech | 해당사항없음 |

한 댓글에 복수 ID가 붙을 수 있습니다 (예: `0,2` = 출신 + 정치). `8`만 있으면 혐오 표현 해당 없음으로 봅니다.

### 이전 샘플과의 차이 (수정 사항)

잘못된 예시(원본과 다름):

- 컬럼 `sentence`, `label`(영문 단일 클래스)
- 클래스별 100건씩 추출·단일 라벨만 남김

현재 샘플은 위 **원본 2컬럼·숫자 라벨** 구조를 그대로 유지합니다.
