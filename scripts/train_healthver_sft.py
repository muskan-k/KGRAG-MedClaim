#!/usr/bin/env python3
import os

# 0) Force offline/local-only mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from datasets import load_dataset
import torch

# 1) Configuration: local model & output directories
MODEL_DIR  = "models/Mistral-7B-Instruct-v0.2"
OUTPUT_DIR = "models/mistral-healthver-lora"

# 2) Ensure offload folder exists (for weight offloading)
os.makedirs("offload", exist_ok=True)

# 3) Load tokenizer from local files only
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    local_files_only=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 4) Load model from local files only, with offloading config
# 4) Load model from local files only, with offloading config
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float32,
    device_map="auto",
    offload_folder="offload",
    trust_remote_code=True,
    local_files_only=True,
    low_cpu_mem_usage=True
)


# 5) Load dataset splits
data = load_dataset(
    "json",
    data_files={
        "train":      "healthver_train.jsonl",
        "validation": "healthver_validation.jsonl"
    }
)
train_ds = data["train"]
val_ds   = data["validation"]

# 6) Data collator for completion-only LM
data_collator = DataCollatorForCompletionOnlyLM(
    tokenizer=tokenizer,
    response_template="JSON:",
    instruction_template=None
)

# 7) Training arguments
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    fp16=torch.cuda.is_available(),
    bf16=False,
    report_to="none"
)

# 8) Prepare LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 9) Initialize SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    peft_config=peft_config
)

# 10) Train & save
trainer.train()
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
