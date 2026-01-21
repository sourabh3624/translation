import torch
from datasets import load_dataset
from transformers import (
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

from config import *

# --------------------------------------------------
# Environment check
# --------------------------------------------------
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
print("Loading dataset...")

dataset = load_dataset(
    "json",
    data_files={
        "train": f"{DATA_DIR}/train.jsonl",
        "test": f"{DATA_DIR}/test.jsonl"
    }
)

# --------------------------------------------------
# Load tokenizer & model
# --------------------------------------------------
tokenizer = MT5Tokenizer.from_pretrained(MODEL_NAME)
model = MT5ForConditionalGeneration.from_pretrained(MODEL_NAME)

if torch.cuda.is_available():
    model.cuda()

# --------------------------------------------------
# Tokenization (NO max_length padding)
# --------------------------------------------------
def tokenize(batch):
    inputs = tokenizer(
        batch["input_text"],
        truncation=True,
        max_length=MAX_LENGTH
    )
    targets = tokenizer(
        batch["target_text"],
        truncation=True,
        max_length=MAX_LENGTH
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

print("Tokenizing dataset...")

tokenized = dataset.map(
    tokenize,
    batched=True,
    remove_columns=dataset["train"].column_names,
    num_proc=NUM_PROC
)

# --------------------------------------------------
# Data collator (dynamic padding + label masking)
# --------------------------------------------------
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100
)

# --------------------------------------------------
# Training arguments (mT5-CORRECT)
# --------------------------------------------------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,

    fp16=False,                      # 

    optim="adafactor",
    learning_rate=1e-3,
    lr_scheduler_type="constant",

    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=SAVE_TOTAL_LIMIT,
    logging_steps=LOGGING_STEPS,
    report_to="none",
    dataloader_num_workers=DATALOADER_WORKERS,
    remove_unused_columns=False
)



# --------------------------------------------------
# Trainer (DEFAULT Trainer — now SAFE)
# --------------------------------------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    data_collator=data_collator
)

# --------------------------------------------------
# Train
# --------------------------------------------------
print(" Starting training (mT5 stable mode)...")
trainer.train()

# --------------------------------------------------
# Save model
# --------------------------------------------------
print(" Saving final model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(" Training complete")
