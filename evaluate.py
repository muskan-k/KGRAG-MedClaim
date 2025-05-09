import json
import os
import argparse
import csv
from collections import Counter
from rag_pipeline import verify_claim
import re

label_map = {
    "REFUTED": "REFUTES",
    "SUPPORTED": "SUPPORTS",
    "NOT ENOUGH INFO": "NEUTRAL"
}

def load_labels_csv(labels_file):
    claims = []
    labels = []
    with open(labels_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            claims.append(row["claim"].strip())
            labels.append(row["label"].strip().upper())
    return claims, labels


def safe_parse_json(output_str):
    try:
        return json.loads(output_str)
    except json.JSONDecodeError:
        match = re.search(r'\{.*?\}', output_str, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise

def evaluate_from_claims(claims, labels):
    y_true = []
    y_pred = []

    for i, claim in enumerate(claims):
        print(f"[INFO] Evaluating claim {i+1}/{len(claims)}")
        gold = labels[i]
        try:
            result = verify_claim(claim)
            llm_output = result["llm_output"]

            if isinstance(llm_output, str):
                llm_output = safe_parse_json(llm_output)

            verdict = llm_output.get("verdict", "").strip().upper()
            mapped_verdict = label_map.get(verdict)

            if mapped_verdict:
                y_pred.append(verdict)
                y_true.append(gold)
            else:
                print(f"[SKIP] No verdict for: {claim}")
        except Exception as e:
            print(f"[WARN] Failed to evaluate: {claim} → {e}")

    if not y_true or not y_pred:
        print("[ERROR] No predictions were collected. Evaluation aborted.")
        return

    count = Counter((t, p) for t, p in zip(y_true, y_pred))
    total = len(y_true)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    acc = correct / total if total else 0.0

    print(f"\nEVALUATION RESULTS ({total} samples):")
    print(f"Accuracy: {acc*100:.2f}%")
    print("Confusion Matrix:")
    labels_set = sorted(set(y_true + y_pred))
    header = [""] + labels_set
    print("\t".join(header))
    for t in labels_set:
        row = [t]
        for p in labels_set:
            row.append(str(count.get((t, p), 0)))
        print("\t".join(row))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline on CSV claims.")
    parser.add_argument("--labels_file", type=str, required=True, help="CSV file with ground-truth labels (columns: claim,label)")
    args = parser.parse_args()

    claims, labels = load_labels_csv(args.labels_file)
    evaluate_from_claims(claims, labels)
