"""
rag_pipeline.py - End‑to‑end vanilla Graph‑RAG claim verifier.
$ python rag_pipeline.py "Metformin increases insulin sensitivity in type 2 diabetes."
"""
import json
from typing import Dict

from query_kg_utils import get_triples_for_claim
from pubmed_fetch import fetch_abstracts
from rag_llm import ask

def format_triples(triples):
    return "\n".join(f"- ({s}, {r}, {o})" for s, r, o in triples) or "None"

def format_abstracts(abs_):
    return "\n".join(f"[{i+1}] {a.replace(chr(10), ' ')}" for i, a in enumerate(abs_)) or "None"

def build_prompt(claim: str, triples, abstracts) -> str:
    return f"""You are a medical claim‑verification assistant.

Knowledge‑Graph evidence:
{format_triples(triples)}

PubMed abstracts:
{format_abstracts(abstracts)}

Task:
1. Output one of the labels: SUPPORTED, REFUTED, NOT ENOUGH INFO.
2. Provide a short justification that cites at least one KG triple and one PubMed sentence (use [index] for abstracts).

Claim: {claim}
Answer (JSON with keys `verdict`, `justification`):
"""

def verify_claim(claim: str, k: int = 20) -> Dict:
    triples = get_triples_for_claim(claim, limit=20)
    abstracts = fetch_abstracts(claim, max_hits=k)
    prompt = build_prompt(claim, triples, abstracts)
    raw = ask(prompt)
    return {
        "claim": claim,
        "triples": triples,
        "pubmed_abstracts": abstracts,
        "llm_output": raw,
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Vanilla KG‑RAG verifier")
    parser.add_argument("claim", nargs="*", help="Claim sentence to verify")
    parser.add_argument("-k", type=int, default=20, help="Max pubmed hits")
    args = parser.parse_args()
    claim = " ".join(args.claim) if args.claim else input("Enter claim: ")
    result = verify_claim(claim, args.k)
    print(json.dumps(result, indent=2))
