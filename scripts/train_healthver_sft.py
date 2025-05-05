# Fine-tunes Mistral-7B-Instruct on HealthVer using QLoRA + TRL

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import load_dataset
import torch

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_DIR = "models/mistral-healthver-lora"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
train_ds = load_dataset("json", data_files="healthver_train.jsonl", split="train")
val_ds   = load_dataset("json", data_files="healthver_validation.jsonl", split="train")

# Collator
collator = DataCollatorForCompletionOnlyLM(
    tokenizer=tokenizer,
    response_template="JSON:",
    instruction_template=None
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,  # use bfloat16 or float16 if available
    device_map="auto"
)

# Training arguments (use from transformers)
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
    fp16=True,  # set to False if no CUDA
    bf16=False,
    report_to="none"
)

# Trainer
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=collator,
    peft_config={
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    },
    max_seq_length=1024,
    dataset_text_field="prompt",
    dataset_label_field="completion",
)

trainer.train()
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
