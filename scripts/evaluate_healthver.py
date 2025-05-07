import json, torch
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score       # ← NEW
from finetuned import load

LABELS = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO"]

def normalize(text: str) -> str:
    """Pull the first word, upper‑case, map to one of the 3 labels."""
    text = text.strip().split()[0].upper()
    if text.startswith("SUPPORT"):
        return "SUPPORTS"
    if text.startswith("REFUTE"):
        return "REFUTES"
    return "NOT_ENOUGH_INFO"

models = {
    "base": load("distilgpt2"),
    "sft":  load("models/distilgpt2-healthver-sft-merged"),   # no '::'
}

with open("data/healthver/test.jsonl") as f:
    data = [json.loads(l) for l in f][:10]   # <= 10‑sample CPU smoke‑test

for tag, (model, tok) in models.items():
    preds, gts = [], []
    for ex in tqdm(data, desc=tag):
        inp = tok(ex["prompt"], return_tensors="pt").to(model.device)
        out = model.generate(**inp, max_new_tokens=30)
        pred = tok.decode(out[0], skip_special_tokens=True)
        preds.append(normalize(pred))
        gts.append(normalize(ex["completion"]))
    acc = accuracy_score(gts, preds)
    f1  = f1_score(gts, preds, labels=LABELS, average="macro")
    print(tag, {"accuracy": acc, "f1_macro": f1})
