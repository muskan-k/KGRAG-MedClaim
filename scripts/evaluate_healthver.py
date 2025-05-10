import json
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter
from finetuned import load

LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]

def normalize_token(tok_str):
    tok_str = tok_str.upper().strip()
    if tok_str.startswith("SUPPORT"):
        return "SUPPORTS"
    if tok_str.startswith("REFUTE"):
        return "REFUTES"
    return "NOT_ENOUGH_INFO"

# Load both base and SFT models
models = {
    # "base": load("distilgpt2"),
    "sft": load("distilgpt2::models/distilgpt2-healthver-sft")
    # "sft":  load("models/distilgpt2-healthver-sft-merged"),
}

# Load test data
with open("data/healthver/test.jsonl") as f:
    data = [json.loads(l) for l in f]  # Use full set or slice if debugging

# Run evaluation
for tag, (model, tok) in models.items():
    preds, gts = [], []

    for ex in tqdm(data, desc=f"Evaluating {tag}"):
        inp = tok(ex["prompt"], return_tensors="pt").to(model.device)
        out = model.generate(
            **inp,
            max_new_tokens=4,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tok.eos_token_id,
        )

        full_output = tok.decode(out[0], skip_special_tokens=True)
        pred_text = full_output.replace(ex["prompt"], "").strip()

        pred = normalize_token(pred_text)
        gold = normalize_token(ex["completion"])

        preds.append(pred)
        gts.append(gold)

    acc = accuracy_score(gts, preds)
    f1  = f1_score(gts, preds, labels=LABELS, average="macro")

    print(f"\n📊 Results for: {tag}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print("Prediction Distribution:", Counter(preds))
    print("Ground Truth Distribution:", Counter(gts))