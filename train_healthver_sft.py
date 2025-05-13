import argparse
import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


def main():
    parser = argparse.ArgumentParser(
        description="Adapter-based fine-tuning on HealthVer CSV data (seq_len=400, GPU-enabled)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        choices=["mistral", "llama2", "biogpt"],
        default="mistral",
        help="Which pretrained checkpoint to fine-tune.",
    )
    args = parser.parse_args()

    MODEL_MAP = {
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "llama2":  "meta-llama/Llama-2-7b-hf",
        "biogpt":  "microsoft/BioGPT-Large-PubMed",
    }
    base_model = MODEL_MAP[args.base_model]

    OUTPUT_DIR = "healthver-sft"
    SEQ_LEN    = 400

    # 1) Clone HealthVer if needed
    repo_dir = "HealthVer"
    if not os.path.isdir(repo_dir):
        os.system(f"git clone https://github.com/sarrouti/HealthVer.git {repo_dir}")

    # 2) Load CSVs via pandas and convert to HF Dataset
    train_csv = os.path.join(repo_dir, "data/train.csv")
    val_csv   = os.path.join(repo_dir, "data/dev.csv")
    train_df  = pd.read_csv(train_csv)
    val_df    = pd.read_csv(val_csv)
    train_ds  = Dataset.from_pandas(train_df)
    val_ds    = Dataset.from_pandas(val_df)

    # 3) Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.model_max_length = SEQ_LEN
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(ex):
        # CSV columns: id,evidence,claim,label,topic_ip,question
        prompt = (
                "Evidence:\n" + ex["evidence"] +
                f"\n\nClaim: {ex['claim']}\n" +
                f"Question: {ex['question']}"
        )
        response = ex["label"]
        text = prompt + "\n### Response: " + response
        return tokenizer(text, truncation=True, max_length=SEQ_LEN)

    train_tok = train_ds.map(
        tokenize_fn,
        batched=False,
        remove_columns=train_ds.column_names
    )
    val_tok = val_ds.map(
        tokenize_fn,
        batched=False,
        remove_columns=val_ds.column_names
    )

    # 4) Load base model and inject LoRA adapters
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    if torch.cuda.is_available():
        model = model.half().to("cuda")
    else:
        model = model.to("cpu")
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()
    for param in model.parameters():
        param.requires_grad = False

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # 5) Trainer setup
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        do_eval=True,
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        per_device_train_batch_size=1 if not torch.cuda.is_available() else 2,
        per_device_eval_batch_size=1 if not torch.cuda.is_available() else 2,
        num_train_epochs=3,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        learning_rate=2e-5,
        warmup_steps=100,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
    )

    # 6) Train and save
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()