# scripts/prepare_healthver.py

import csv
import json
import pathlib

# adjust this path to where your CSVs actually live:
INPUT_DIR = pathlib.Path("data/csv")
OUTPUT_DIR = pathlib.Path("data/healthver")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# map CSV filenames → output splits
splits = {
    "train":      "healthver_train.csv",
    "validation": "healthver_dev.csv",
    "test":       "healthver_test.csv",
}

# map possible CSV labels (case‑insensitive) → target texts
LABEL2TXT = {
    "supports":        "SUPPORTED",
    "refutes":         "REFUTED",
    "neutral":         "NOT ENOUGH INFO",
    "not enough info": "NOT ENOUGH INFO",
    "nei":             "NOT ENOUGH INFO",
}

def normalize_label(raw_label: str) -> str:
    key = raw_label.strip().lower()
    if key not in LABEL2TXT:
        raise ValueError(f"Unrecognized label: {raw_label!r}")
    return LABEL2TXT[key]

for split, fname in splits.items():
    in_path  = INPUT_DIR / fname
    out_path = OUTPUT_DIR / f"{split}.jsonl"

    count = 0
    with open(in_path, newline="", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        for row in reader:
            claim = row["claim"]

            # parse evidence: try JSON list, else semicolon‑split, else single string
            raw_ev = row.get("evidence", "")
            try:
                evidence_list = json.loads(raw_ev)
            except (json.JSONDecodeError, TypeError):
                evidence_list = [s.strip() for s in raw_ev.split(";") if s.strip()]

            # normalize your label names (Supports/Refutes/Neutral/etc)
            label_txt = normalize_label(row["label"])

            # build the SFT prompt/completion
            prompt     = f"Claim: {claim}\nEvidence: {' '.join(evidence_list)}\nAnswer:"
            completion = f" {label_txt}"  # leading space is required by HF SFTTrainer

            fout.write(json.dumps({
                "prompt":     prompt,
                "completion": completion
            }) + "\n")
            count += 1

    print(f"Wrote {count} records for split={split} → {out_path}")
