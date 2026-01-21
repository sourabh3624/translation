import torch
from datasets import load_dataset
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from jiwer import cer
from tqdm import tqdm

from config import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = MT5Tokenizer.from_pretrained(OUTPUT_DIR)
model = MT5ForConditionalGeneration.from_pretrained(OUTPUT_DIR).to(DEVICE)
model.eval()


dataset = load_dataset(
    "json",
    data_files={"test": f"{DATA_DIR}/test.jsonl"}
)["test"]

def transliterate(text):
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_LENGTH,
            num_beams=4
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

total_cer = 0.0
exact_matches = 0


for sample in tqdm(dataset.select(range(5000))):  # evaluate on 5k samples
    pred = transliterate(sample["input_text"])
    tgt = sample["target_text"]

    total_cer += cer(tgt, pred)
    if pred == tgt:
        exact_matches += 1

num_samples = 5000
avg_cer = total_cer / num_samples
accuracy = exact_matches / num_samples


print(f"Character Error Rate (CER): {avg_cer:.4f}")
print(f"Exact Match Accuracy     : {accuracy:.4f}")
