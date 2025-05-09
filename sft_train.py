from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments
)
import datasets

# 1) Load model & tokenizer
MODEL = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    low_cpu_mem_usage=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)

# 2) Load your data
data = datasets.load_dataset("json", data_files={
    "train": "healthver_train.jsonl",
    "validation": "healthver_validation.jsonl"
})

# 3) Tokenization function
def preprocess(batch):
    enc = tokenizer(
        batch["prompt"],
        truncation=True,
        padding="longest",
        max_length=1024
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["completion"],
            truncation=True,
            padding="longest",
            max_length=512
        )
    enc["labels"] = labels["input_ids"]
    return enc

data = data.map(preprocess, batched=True, remove_columns=data["train"].column_names)

# 4) Training args
args = TrainingArguments(
    output_dir="models/healthver-sft",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    learning_rate=2e-5,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    save_total_limit=2,
)

# 5) Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=data["train"],
    eval_dataset=data["validation"],
    tokenizer=tokenizer,
)

# 6) Run!
trainer.train()
trainer.save_model()
