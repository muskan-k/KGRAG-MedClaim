"""
rag_pipeline.py - End-to-end Graph-RAG medical claim verifier (fine-tuned & instruction-tuned model).
Usage:
  python rag_pipeline.py "Metformin increases insulin sensitivity in type 2 diabetes."
"""
import os
import json
from typing import Dict, List, Tuple

from query_kg_utils import get_triples_for_claim
from pubmed_fetch import get_relevant_abstracts
from finetuned import load as load_model

# Load the fine-tuned (and instruction-tuned) model
MODEL_PATH = os.getenv("LLM_PATH", "models/llama2-healthver-sft-it")
model, tokenizer = load_model(MODEL_PATH)


def format_triples(triples: List[Tuple[str, str, str]]) -> str:
    return "\n".join(f"- ({s}, {r}, {o})" for s, r, o in triples) or "None"


def format_abstracts(abstracts: List[str]) -> str:
    return "\n".join(f"[{i+1}] {txt.replace(chr(10), ' ')}"
                         for i, txt in enumerate(abstracts)) or "None"


def build_prompt(claim: str,
                 triples: List[Tuple[str, str, str]],
                 abstracts: List[str]) -> str:
    kg_chunks = "\n".join(f"[KG {i+1}] ({s}, {r}, {o})"
                             for i, (s, r, o) in enumerate(triples))
    pm_chunks = "\n".join(f"[PM {i+1}] {txt.replace(chr(10), ' ')}"
                             for i, txt in enumerate(abstracts))
    return (
        "You are a medical fact-checker.\n"
        f"Claim: {claim}\n\n"
        "KG Evidence:\n"
        f"{kg_chunks or 'None'}\n\n"
        "PubMed Evidence:\n"
        f"{pm_chunks or 'None'}\n\n"
        "Answer with one of [SUPPORTED, REFUTED, NOT ENOUGH INFO] "
        "and then a one-sentence justification citing at least one KG and one PubMed item.\n\n"
        "Respond in JSON with keys: `verdict`, `justification`.\n"
    )


class LLMClient:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=128)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# Instantiate a reusable LLM client
llm = LLMClient(model, tokenizer)


def verify_claim(claim: str, k: int = 20) -> Dict:
    """
    Retrieves KG triples and PubMed abstracts for `claim`, builds a prompt,
    and generates a JSON verdict+justification from the fine-tuned LLM.
    """
    triples = get_triples_for_claim(claim, limit=k)
    abstracts = get_relevant_abstracts(claim, top_k=k)
    prompt = build_prompt(claim, triples, abstracts)
    llm_output = llm(prompt)

    return {
        "claim": claim,
        "triples": triples,
        "pubmed_abstracts": abstracts,
        "llm_output": llm_output,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Graph-RAG medical claim verifier")
    parser.add_argument(
        "claim", nargs="*", help="Claim sentence to verify"
    )
    parser.add_argument(
        "-k", type=int, default=5,
        help="How many PubMed abstracts to retrieve after BM25"
    )
    args = parser.parse_args()
    claim_text = " ".join(args.claim) if args.claim else input("Enter claim: ")
    result = verify_claim(claim_text, args.k)
    print(json.dumps(result, indent=2))
