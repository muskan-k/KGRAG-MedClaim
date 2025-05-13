# train_healthver_sft.py

import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a causal-LM on HealthVer data (seq_len=400, GPU-enabled)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        choices=["mistral", "llama2", "biogpt"],
        default="mistral",
        help="Which pre-trained checkpoint to fine-tune (no GPT-2).",
    )
    args = parser.parse_args()

    # Map shorthand to actual HF model IDs
    MODEL_MAP = {
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "llama2":  "meta-llama/Llama-2-7b-hf",
        "biogpt":  "microsoft/BioGPT-Large-PubMed",
    }
    base_model = MODEL_MAP[args.base_model]

    OUTPUT_DIR = "healthver-sft"
    SEQ_LEN    = 400

    # 1) Load HealthVer JSONL train/validation splits
    data_files = {
        "train":      "data/healthver_train.jsonl",
        "validation": "data/healthver_val.jsonl",
    }
    train_ds, val_ds = load_dataset(
        "json", data_files=data_files, split=["train", "validation"]
    )

    # 2) Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    tokenizer.model_max_length = SEQ_LEN

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(ex):
        text = ex["prompt"] + ex["response"]
        return tokenizer(
            text,
            truncation=True,
            max_length=SEQ_LEN,
            padding=False,
        )

    train_tok = train_ds.map(
        tokenize_fn, batched=False, remove_columns=train_ds.column_names
    )
    val_tok = val_ds.map(
        tokenize_fn, batched=False, remove_columns=val_ds.column_names
    )

    model = AutoModelForCausalLM.from_pretrained(base_model)
    model.resize_token_embeddings(len(tokenizer))

    # 3) Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # 4) Trainer arguments (GPU-enabled)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        do_eval=True,
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        fp16=True,       # use mixed precision on GPU
        # no_cuda removed to allow GPU
        save_total_limit=2,
        learning_rate=2e-5,
        warmup_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
    )

    # 5) Train & save
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()