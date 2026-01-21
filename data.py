import os
import zipfile
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ---------------- CONFIG ---------------- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(BASE_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

ZIP_FILES = {
    "hi": "hin.zip",
    "bn": "ben.zip",
    "ta": "tam.zip"
}

FILE_MAP = {
    "hi": ("hin_train.json", "hin_valid.json", "hin_test.json"),
    "bn": ("ben_train.json", "ben_valid.json", "ben_test.json"),
    "ta": ("tam_train.json", "tam_valid.json", "tam_test.json"),
}

TRAIN_SPLIT = 0.9
RANDOM_STATE = 42

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ---------------------------------------- #


def extract_zip(zip_path, extract_to):
    os.makedirs(extract_to, exist_ok=True)
    if os.listdir(extract_to):
        print(f"Already extracted: {extract_to}")
        return
    print(f"Extracting {os.path.basename(zip_path)} → {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)


def get_src_tgt(obj, lang):
    """
    Explicit support for ALL known Aksharantar formats
    """

    # Your exact format (Dakshina)
    if "english word" in obj and "native word" in obj:
        return obj["english word"], obj["native word"]

    # Common Aksharantar variants
    if "src" in obj and "tgt" in obj:
        return obj["src"], obj["tgt"]

    if "roman" in obj and "native" in obj:
        return obj["roman"], obj["native"]

    if "input" in obj and "output" in obj:
        return obj["input"], obj["output"]

    if "en" in obj and lang in obj:
        return obj["en"], obj[lang]

    raise KeyError(f" Unsupported format: {obj}")


def load_aksharantar_file(path, lang):
    pairs = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            try:
                src, tgt = get_src_tgt(obj, lang)
            except KeyError as e:
                print(f"Skipping line {line_no}: {e}")
                continue

            src, tgt = src.strip(), tgt.strip()
            if src and tgt:
                pairs.append((src, tgt))

    return pairs


def load_language(lang):
    zip_path = os.path.join(BASE_DIR, ZIP_FILES[lang])
    extract_path = os.path.join(RAW_DIR, lang)

    extract_zip(zip_path, extract_path)

    train_f, valid_f, test_f = FILE_MAP[lang]

    all_pairs = []
    for fname in (train_f, valid_f, test_f):
        path = os.path.join(extract_path, fname)
        print(f"Loading: {path}")
        all_pairs.extend(load_aksharantar_file(path, lang))

    samples = []
    for src, tgt in tqdm(all_pairs, desc=f"Processing {lang}"):
        samples.append({
            "input_text": f"<{lang}> {src}",
            "target_text": tgt,
            "lang": lang
        })

    return samples


def main():
    all_samples = []

    for lang in ZIP_FILES:
        print(f"\n Language: {lang}")
        samples = load_language(lang)
        print(f"Loaded {len(samples)} samples")
        all_samples.extend(samples)

    if not all_samples:
        raise RuntimeError("No samples loaded — dataset parsing failed")

    print(f"\n Total samples: {len(all_samples)}")

    train_data, test_data = train_test_split(
        all_samples,
        test_size=1 - TRAIN_SPLIT,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    train_path = os.path.join(PROCESSED_DIR, "train.jsonl")
    test_path = os.path.join(PROCESSED_DIR, "test.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for row in train_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(test_path, "w", encoding="utf-8") as f:
        for row in test_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n Data preparation complete")
    print(f" Train: {train_path}")
    print(f" Test : {test_path}")


if __name__ == "__main__":
    main()
