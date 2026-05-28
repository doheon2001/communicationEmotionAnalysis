"""
기능1 공통 데이터셋 빌더 — emotion_balanced_v1

규칙:
  - 소스: AIHub 감성대화 말뭉치 (Training + Validation xlsx)
  - 텍스트: 사람문장1
  - 라벨: 기쁨=1, 나머지(슬픔/분노/불안/상처/당황)=0
  - 균형 샘플: 긍정 5000 + 부정 5000 = 10000
  - 분할: train 80% / valid 10% / test 10% (stratified, seed 42)
  - 전처리: NaN 제거 + 양끝 공백 strip + 빈 문자열 제거 (그 외 원문 유지)

산출물 (모두 data/processed/ 아래):
  emotion_balanced_v1.csv          전체 10000
  emotion_balanced_v1_train.csv    8000
  emotion_balanced_v1_valid.csv    1000
  emotion_balanced_v1_test.csv     1000
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
N_PER_CLASS = 5000
POS_EMOTION = "기쁨"

REPO_ROOT = Path(__file__).resolve().parent
CHAEON_ROOT = REPO_ROOT.parent
SRC_TRAIN = CHAEON_ROOT / "AIHub 감성 대화 말뭉치" / "Training_221115_add" / "원천데이터" / "감성대화말뭉치(최종데이터)_Training.xlsx"
SRC_VAL   = CHAEON_ROOT / "AIHub 감성 대화 말뭉치" / "Validation_221115_add" / "원천데이터" / "감성대화말뭉치(최종데이터)_Validation.xlsx"
OUT_DIR   = REPO_ROOT / "data" / "processed"


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)[["사람문장1", "감정_대분류"]]
    df = df.dropna(subset=["사람문장1", "감정_대분류"]).copy()
    df["사람문장1"] = df["사람문장1"].astype(str).str.strip()
    df = df[df["사람문장1"] != ""]
    return df


def main() -> None:
    print(f"[load] {SRC_TRAIN.name}")
    train_src = load_and_clean(SRC_TRAIN)
    print(f"[load] {SRC_VAL.name}")
    val_src = load_and_clean(SRC_VAL)

    pool = pd.concat([train_src, val_src], ignore_index=True)
    print(f"\n[pool] 합친 원천 행수: {len(pool):,}")
    print(pool["감정_대분류"].value_counts().to_string())

    pool["label"] = (pool["감정_대분류"] == POS_EMOTION).astype(int)
    pos = pool[pool["label"] == 1]
    neg = pool[pool["label"] == 0]
    print(f"\n[pool] 긍정(기쁨) {len(pos):,} / 부정 {len(neg):,}")

    assert len(pos) >= N_PER_CLASS, f"긍정 부족: {len(pos)} < {N_PER_CLASS}"
    assert len(neg) >= N_PER_CLASS, f"부정 부족: {len(neg)} < {N_PER_CLASS}"

    pos_s = pos.sample(n=N_PER_CLASS, random_state=SEED)
    neg_s = neg.sample(n=N_PER_CLASS, random_state=SEED)
    balanced = (
        pd.concat([pos_s, neg_s])
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
        .rename(columns={"사람문장1": "text"})[["text", "label"]]
    )
    print(f"\n[balanced] {len(balanced):,}행, 라벨 분포:")
    print(balanced["label"].value_counts().to_string())

    train_df, tmp_df = train_test_split(
        balanced, test_size=0.20, random_state=SEED, stratify=balanced["label"]
    )
    valid_df, test_df = train_test_split(
        tmp_df, test_size=0.50, random_state=SEED, stratify=tmp_df["label"]
    )
    for name, d in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        d.reset_index(drop=True, inplace=True)
        print(f"  {name}: {len(d):,}  (pos {int(d['label'].sum())} / neg {int((1-d['label']).sum())})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "emotion_balanced_v1.csv": balanced,
        "emotion_balanced_v1_train.csv": train_df,
        "emotion_balanced_v1_valid.csv": valid_df,
        "emotion_balanced_v1_test.csv": test_df,
    }
    print(f"\n[write] -> {OUT_DIR}")
    for fname, d in paths.items():
        out = OUT_DIR / fname
        d.to_csv(out, index=False, encoding="utf-8")
        print(f"  {fname}  ({len(d):,}행)")


if __name__ == "__main__":
    main()
