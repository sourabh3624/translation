# ===============================
# Model
# ===============================
MODEL_NAME = "google/mt5-small"

# ===============================
# Paths
# ===============================
DATA_DIR = "../data/processed"
OUTPUT_DIR = "../models/mt5-translit"

# ===============================
# Languages (documentation)
# ===============================
LANGUAGES = ["hi", "bn", "ta"]

# ===============================
# Tokenization
# ===============================
MAX_LENGTH = 64   # sufficient for transliteration

# ===============================
# Training (V100 32GB OPTIMAL)
# ===============================
BATCH_SIZE = 64              # fits comfortably on V100 with fp16
GRAD_ACCUM_STEPS = 1
LEARNING_RATE = 1e-4         # ↓ reduced for FP16 stability
EPOCHS = 1                   # large dataset → 1 epoch enough
FP16 = True

# ===============================
# Performance
# ===============================
NUM_PROC = 8                 # CPU workers for tokenization
DATALOADER_WORKERS = 4
LOGGING_STEPS = 1000
SAVE_TOTAL_LIMIT = 2
MAX_GRAD_NORM = 1.0         
