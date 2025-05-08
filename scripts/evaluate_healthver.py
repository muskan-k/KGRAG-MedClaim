import json, torch
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score       # ← NEW
from finetuned import load

LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]

def normalize_token(tok_str):
    tok_str = tok_str.upper()
    if tok_str.startswith("SUPPORT"):
        return "SUPPORTS"
    if tok_str.startswith("REFUTE"):
        return "REFUTES"
    return "NOT_ENOUGH_INFO"

models = {
    "base": load("distilgpt2"),
    "sft":  load("models/distilgpt2-healthver-sft-merged"),   # no '::'
}

with open("data/healthver/test.jsonl") as f:
    data = [json.loads(l) for l in f][:500]   # <= 10‑sample CPU smoke‑test

for tag, (model, tok) in models.items():
    preds, gts = [], []
    for ex in tqdm(data, desc=tag):
        inp = tok(ex["prompt"], return_tensors="pt").to(model.device)
        out = model.generate(
            **inp,
            max_new_tokens=1,
            do_sample=False,
            temperature=0.0,
        )
        pred = tok.decode(out[0], skip_special_tokens=True)
        preds.append(normalize_token(pred))
        gts.append(normalize_token(ex["completion"]))
    acc = accuracy_score(gts, preds)
    f1  = f1_score(gts, preds, labels=LABELS, average="macro")
    print(tag, {"accuracy": acc, "f1_macro": f1})
