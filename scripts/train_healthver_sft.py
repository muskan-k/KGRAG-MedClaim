# Full-parameter SFT of Falcon-7B on HealthVer data

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

# 1) Configuration
# Use a publicly available, non-gated model
MODEL_NAME = "tiiuae/falcon-7b"
OUTPUT_DIR = "models/falcon7b-healthver-sft"

# 2) Load tokenizer & model
# Falcon-7B uses byte-level BPE tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
# Ensure padding token is defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# 3) Load & preprocess dataset
# Expect JSONL with fields: 'prompt', 'completion'
data = load_dataset(
    "json",
    data_files={
        "train": "healthver_train.jsonl",
        "validation": "healthver_validation.jsonl"
    }
)

def preprocess(batch):
    # Tokenize prompts
    enc = tokenizer(
        batch["prompt"],
        truncation=True,
        padding="longest",
        max_length=1024
    )
    # Tokenize completions as labels
    with tokenizer.as_target_tokenizer():
        lbl = tokenizer(
            batch["completion"],
            truncation=True,
            padding="longest",
            max_length=512
        )
    enc["labels"] = lbl["input_ids"]
    return enc

# Apply preprocessing remove original columns
data = data.map(
    preprocess,
    batched=True,
    remove_columns=data["train"].column_names
)
train_ds = data["train"]
val_ds   = data["validation"]

# 4) Data collator for causal LM
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 5) Training arguments
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    learning_rate=2e-5,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    fp16=False,        # CPU or single-GPU
    report_to="none"
)

# 6) Trainer setup
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 7) Run training and save
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
