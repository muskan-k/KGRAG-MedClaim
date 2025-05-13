import os
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

# --- Configuration -----------------------------------------------------
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_DIR = "monant-instr-sft"
SEQ_LEN    = 512

# --- Clone Monant sample_data if needed --------------------------------
REPO_DIR = "medical-misinformation-dataset"
if not os.path.isdir(REPO_DIR):
    os.system(f"git clone https://github.com/kinit-sk/medical-misinformation-dataset.git {REPO_DIR}")

# --- Load sample CSVs into pandas --------------------------------------
articles = pd.read_csv(os.path.join(REPO_DIR, "sample_data/articles.csv"))
claims   = pd.read_csv(os.path.join(REPO_DIR, "sample_data/claims.csv"))
facts    = pd.read_csv(os.path.join(REPO_DIR, "sample_data/fact_checking_articles.csv"))

# --- Merge into a single DataFrame -------------------------------------
# facts: claim_id, article_id, claim_present, stance
# claims: id, claim_text
# articles: id, content
merged = (
    facts.rename(columns={'claim_id': 'cid', 'article_id': 'aid'})
    .merge(claims.rename(columns={'id': 'cid', 'claim': 'claim_text'}), on='cid')
    .merge(articles.rename(columns={'id': 'aid', 'content': 'article_text'}), on='aid')
)

# --- Sample and split --------------------------------------------------
# use a small sample for instruction tuning
sampled = merged.sample(n=min(2000, len(merged)), random_state=42).reset_index(drop=True)

split = int(0.8 * len(sampled))
train_df = sampled.iloc[:split]
val_df   = sampled.iloc[split:]

# --- Convert to HuggingFace Dataset ------------------------------------
datasets = DatasetDict({
    'train': Dataset.from_pandas(train_df),
    'validation': Dataset.from_pandas(val_df)
})

# --- Tokenizer ----------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_fn(ex):
    instruction = (
        "You are a medical fact-checker. "
        "Given an article excerpt and a claim, determine whether the claim is SUPPORTED, CONTRADICTED, or NEUTRAL, "
        "and explain in one sentence."
    )
    prompt = (
            instruction + "\n\n"
                          f"Article Excerpt:\n{ex['article_text']}\n\n"
                          f"Claim: {ex['claim_text']}\n"
                          "Question: Is the claim SUPPORTED, CONTRADICTED, or NEUTRAL?\n"
                          "### Response: "
    )
    # Use stance as the label plus short explanation
    response = ex['stance']
    text = prompt + response
    return tokenizer(text, truncation=True, max_length=SEQ_LEN)

# Map datasets
tokenized = datasets.map(tokenize_fn, batched=False, remove_columns=datasets['train'].column_names)

# --- Model + LoRA injection --------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
)
if torch.cuda.is_available():
    model = model.half().to('cuda')
else:
    model = model.to('cpu')
model.resize_token_embeddings(len(tokenizer))
model.gradient_checkpointing_enable()
for param in model.parameters():
    param.requires_grad = False

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=['q_proj', 'v_proj'],
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM',
)
model = get_peft_model(model, lora_cfg)

# --- Training setup ----------------------------------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    evaluation_strategy='steps',
    eval_steps=200,
    save_steps=200,
    logging_steps=100,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    fp16=torch.cuda.is_available(),
    gradient_checkpointing=True,
    learning_rate=5e-5,
    warmup_steps=50,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['validation'],
    data_collator=data_collator,
)

# --- Run training -------------------------------------------------------
if __name__ == '__main__':
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
