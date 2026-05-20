"""
Generate raw sample CSVs in data/raw/ from local AIHub / K-HATERS / K-MHaS sources.

Outputs mirror original row/column structure (no extra metadata columns).
See data/raw/README.md for sampling notes and format details.
"""

from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path

import pandas as pd

WORKSPACE = Path(__file__).resolve().parent
RAW_DIR = WORKSPACE / "data" / "raw"

EMOTION_XLSX_NAME = "감성대화말뭉치(최종데이터)_Training.xlsx"
K_HATERS_MERGED = RAW_DIR / "Selectstar_Tunip_HUMANELab_opendata" / "Selectstar_Tunip_HUMANELab_opendata_merged.json"
K_MHAS_TRAIN_URL = "https://raw.githubusercontent.com/adlnlp/K-MHaS/main/data/kmhas_train.txt"
K_MHAS_TRAIN_LOCAL = RAW_DIR / "K-MHaS" / "kmhas_train.txt"
K_MHAS_COLUMNS = ["document", "label"]

EMOTION_COLUMNS = [
    "Unnamed: 0",
    "연령",
    "성별",
    "상황키워드",
    "전체선톡",
    "감정_대분류",
    "감정_소분류",
    "사람문장1",
    "시스템문장1",
    "사람문장2",
    "시스템문장2",
    "사람문장3",
    "시스템문장3",
]

RANDOM_STATE = 42


def find_desktop_source_dir() -> Path:
    desktop = Path.home() / "Desktop"
    for entry in desktop.iterdir():
        if not entry.is_dir():
            continue
        if (entry / "talksets-train-1.txt").exists():
            return entry
    raise FileNotFoundError(
        "Could not find Desktop folder containing talksets-train-1.txt"
    )


def read_text_utf8(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def ensure_utf8_copy(src: Path, dest: Path) -> None:
    """Re-encode source file as UTF-8 under data/raw/."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in {".xlsx", ".xls"}:
        dest.write_bytes(src.read_bytes())
        return
    text = read_text_utf8(src)
    dest.write_text(text, encoding="utf-8", newline="\n")


def load_emotion_dataframe(source_dir: Path) -> pd.DataFrame:
    xlsx_path = source_dir / EMOTION_XLSX_NAME
    if not xlsx_path.exists():
        matches = list(source_dir.glob("*Training.xlsx"))
        if not matches:
            raise FileNotFoundError(f"Emotion XLSX not found in {source_dir}")
        xlsx_path = matches[0]

    local_xlsx = RAW_DIR / "018.감성대화" / "Training_221115_add" / "원천데이터" / EMOTION_XLSX_NAME
    ensure_utf8_copy(xlsx_path, local_xlsx)

    df = pd.read_excel(local_xlsx)
    if len(df.columns) == len(EMOTION_COLUMNS):
        df.columns = EMOTION_COLUMNS
    return df


def sample_emotion(df: pd.DataFrame, n: int = 300) -> pd.DataFrame:
    sampled = df.sample(n=min(n, len(df)), random_state=RANDOM_STATE)
    return sampled.reset_index(drop=True)


def load_k_haters_dataframe() -> pd.DataFrame:
    if not K_HATERS_MERGED.exists():
        raise FileNotFoundError(f"K-HATERS merged JSON not found: {K_HATERS_MERGED}")

    with K_HATERS_MERGED.open("r", encoding="utf-8") as f:
        records = json.load(f)
    return pd.DataFrame(records)


def sample_k_haters(df: pd.DataFrame, n: int = 400) -> pd.DataFrame:
    sampled = df.sample(n=min(n, len(df)), random_state=RANDOM_STATE)
    return sampled.reset_index(drop=True)


def load_talkset_lines(source_dir: Path) -> list[str]:
    """Load talksets-train-1~5.txt lines (one talkset per line, same as original txt)."""
    lines: list[str] = []
    for part in range(1, 6):
        txt_name = f"talksets-train-{part}.txt"
        txt_path = source_dir / txt_name
        if not txt_path.exists():
            raise FileNotFoundError(txt_path)

        local_txt = RAW_DIR / "talksets" / txt_name
        ensure_utf8_copy(txt_path, local_txt)
        lines.extend(read_text_utf8(local_txt).splitlines())
    return lines


def sample_talkset_lines(lines: list[str], n: int = 200) -> list[str]:
    series = pd.Series(lines, dtype="string")
    sampled = series.sample(n=min(n, len(series)), random_state=RANDOM_STATE)
    return sampled.tolist()


def ensure_k_mhas_train_file() -> Path:
    if K_MHAS_TRAIN_LOCAL.exists():
        return K_MHAS_TRAIN_LOCAL

    K_MHAS_TRAIN_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading K-MHaS train file to {K_MHAS_TRAIN_LOCAL} ...")
    req = urllib.request.Request(K_MHAS_TRAIN_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        content = response.read().decode("utf-8")
    K_MHAS_TRAIN_LOCAL.write_text(content, encoding="utf-8", newline="\n")
    return K_MHAS_TRAIN_LOCAL


def load_k_mhas_dataframe() -> pd.DataFrame:
    """Load kmhas_train.txt: tab-separated document + comma-separated label IDs."""
    train_path = ensure_k_mhas_train_file()
    text = read_text_utf8(train_path)
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"Empty K-MHaS train file: {train_path}")

    header = lines[0].split("\t")
    if header != K_MHAS_COLUMNS:
        raise ValueError(
            f"Unexpected K-MHaS header {header!r}, expected {K_MHAS_COLUMNS}"
        )

    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        document, label = line.split("\t", 1)
        rows.append({"document": document, "label": label.strip()})

    return pd.DataFrame(rows, columns=K_MHAS_COLUMNS)


def sample_k_mhas(df: pd.DataFrame, n: int = 400) -> pd.DataFrame:
    sampled = df.sample(n=min(n, len(df)), random_state=RANDOM_STATE)
    return sampled.reset_index(drop=True)


def write_talkset_csv(lines: list[str], out_path: Path) -> None:
    """Write like original txt: one field per row, no header row."""
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        for line in lines:
            writer.writerow([line])


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    source_dir = find_desktop_source_dir()
    print(f"Source directory: {source_dir}")

    print("Generating aihub_emotion_raw_sample.csv ...")
    emotion_df = load_emotion_dataframe(source_dir)
    emotion_sample = sample_emotion(emotion_df, 300)
    emotion_out = RAW_DIR / "aihub_emotion_raw_sample.csv"
    emotion_sample.to_csv(emotion_out, index=False, encoding="utf-8-sig")
    print(f"  -> {emotion_out} ({len(emotion_sample)} rows, {len(emotion_sample.columns)} cols)")

    print("Generating k_haters_raw_sample.csv ...")
    k_haters_df = load_k_haters_dataframe()
    k_haters_sample = sample_k_haters(k_haters_df, 400)
    k_haters_out = RAW_DIR / "k_haters_raw_sample.csv"
    k_haters_sample.to_csv(k_haters_out, index=False, encoding="utf-8-sig")
    print(f"  -> {k_haters_out} ({len(k_haters_sample)} rows, {len(k_haters_sample.columns)} cols)")

    print("Generating aihub_talkset_raw_sample.csv ...")
    talkset_lines = load_talkset_lines(source_dir)
    talkset_sample = sample_talkset_lines(talkset_lines, 200)
    talkset_out = RAW_DIR / "aihub_talkset_raw_sample.csv"
    write_talkset_csv(talkset_sample, talkset_out)
    print(f"  -> {talkset_out} ({len(talkset_sample)} rows, no header, 1 col)")

    print("Generating k_mhas_raw_sample.csv ...")
    k_mhas_df = load_k_mhas_dataframe()
    k_mhas_sample = sample_k_mhas(k_mhas_df, 400)
    k_mhas_out = RAW_DIR / "k_mhas_raw_sample.csv"
    k_mhas_sample.to_csv(k_mhas_out, index=False, encoding="utf-8-sig")
    print(f"  -> {k_mhas_out} ({len(k_mhas_sample)} rows, {len(k_mhas_sample.columns)} cols)")

    print("Done.")


if __name__ == "__main__":
    main()
