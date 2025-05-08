#!/usr/bin/env python3
import os
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model

# ─────────────────────────────────────────────────────────────────────────────
# 0) CONFIG: how many batches per epoch
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE        = 50 if torch.cuda.is_available() else 1
BATCHES_PER_EPOCH = 20
MAX_TRAIN_EXAMPLES = BATCH_SIZE * BATCHES_PER_EPOCH
MAX_EVAL_EXAMPLES  = BATCH_SIZE * BATCHES_PER_EPOCH

# ─────────────────────────────────────────────────────────────────────────────
# 1) Paths & base model
# ─────────────────────────────────────────────────────────────────────────────
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
MODEL_NAME = "distilgpt2"
DATA_DIR   = Path("data/healthver")
OUT_DIR    = Path("models/distilgpt2-healthver-sft")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2) Load & quantize base model
# ─────────────────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
else:
    dtype = torch.float16 if torch.backends.mps.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# 3) Tokenizer
# ─────────────────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# ─────────────────────────────────────────────────────────────────────────────
# 4) Attach LoRA adapter
# ─────────────────────────────────────────────────────────────────────────────
peft_cfg = LoraConfig(
    r=4,
    lora_alpha=32,
    task_type="CAUSAL_LM",
    inference_mode=False,
)
model = get_peft_model(base_model, peft_cfg)

# ─────────────────────────────────────────────────────────────────────────────
# 5) Load & preprocess dataset
# ─────────────────────────────────────────────────────────────────────────────
raw = load_dataset(
    "json",
    data_files={
        "train":      str(DATA_DIR / "train.jsonl"),
        "validation": str(DATA_DIR / "validation.jsonl"),
    }
)

def preprocess(ex):
    prompt, completion = ex["prompt"], ex["completion"]
    enc = tokenizer(
        prompt + completion,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    # mask out prompt portion
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    labels     = input_ids.copy()
    for i in range(len(prompt_ids)):
        labels[i] = -100
    return {"input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels}

tok = raw.map(preprocess,
              remove_columns=raw["train"].column_names,
              batched=False)

# slice only the first N examples so you get exactly 10 batches
small_train = tok["train"].select(range(min(MAX_TRAIN_EXAMPLES, len(tok["train"]))))
small_val   = tok["validation"].select(range(min(MAX_EVAL_EXAMPLES,  len(tok["validation"]))))

# ─────────────────────────────────────────────────────────────────────────────
# 6) TrainingArguments (no unsupported args here)
# ─────────────────────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=str(OUT_DIR),
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=64,
    num_train_epochs=10,
    learning_rate=2e-4,
    logging_steps=20,
    save_steps=500,
    fp16=torch.cuda.is_available(),
    push_to_hub=False,
    # if you’d rather stop by total steps, uncomment:
    # max_steps=100,
)

# ─────────────────────────────────────────────────────────────────────────────
# 7) Trainer init & train
# ─────────────────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train,
    eval_dataset= small_val,
    data_collator=default_data_collator,
)

trainer.train()

# ─────────────────────────────────────────────────────────────────────────────
# 8) Save adapter & tokenizer
# ─────────────────────────────────────────────────────────────────────────────
model.save_pretrained(str(OUT_DIR))
tokenizer.save_pretrained(str(OUT_DIR))
