import time
import torch
import ctranslate2
from transformers import MT5Tokenizer, MT5ForConditionalGeneration

# ----------------------------
# Paths
# ----------------------------
PT_MODEL = "../models/mt5-translit"
CT2_MODEL = "../models/ct2-mt5"

# ----------------------------
# Sample inputs
# ----------------------------
SAMPLES = [
    "<hi> janamdivas",
    "<bn> bhalobasha",
    "<ta> vanakkam"
]

N_RUNS = 100

# ----------------------------
# Load tokenizer
# ----------------------------
tokenizer = MT5Tokenizer.from_pretrained(PT_MODEL)

# =========================================================
# PyTorch benchmark
# =========================================================
print("\n PyTorch mT5 Benchmark")

pt_model = MT5ForConditionalGeneration.from_pretrained(PT_MODEL).cuda()
pt_model.eval()

torch.cuda.synchronize()
start = time.time()

for _ in range(N_RUNS):
    for text in SAMPLES:
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            pt_model.generate(**inputs, max_length=32)

torch.cuda.synchronize()
pt_time = time.time() - start

print(f"PyTorch total time: {pt_time:.2f}s")
print(f"Avg per request: {pt_time / (N_RUNS * len(SAMPLES)):.4f}s")

# =========================================================
# CTranslate2 benchmark
# =========================================================
print("\n⚡ CTranslate2 Benchmark")

translator = ctranslate2.Translator(
    CT2_MODEL,
    device="cuda",
    compute_type="int8"
)

start = time.time()

for _ in range(N_RUNS):
    for text in SAMPLES:
        tokens = tokenizer.convert_ids_to_tokens(
            tokenizer(text)["input_ids"]
        )
        translator.translate_batch(
            [tokens],
            max_decoding_length=32
        )

ct2_time = time.time() - start

print(f"CTranslate2 total time: {ct2_time:.2f}s")
print(f"Avg per request: {ct2_time / (N_RUNS * len(SAMPLES)):.4f}s")

# =========================================================
# Speedup
# =========================================================
speedup = pt_time / ct2_time
print(f"\n Speedup: {speedup:.2f}×")
