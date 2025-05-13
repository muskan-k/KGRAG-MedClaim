import argparse
from biogpt import verify_claim as use_biogpt
from llama2 import verify_claim as use_llama2
from mistral import verify_claim as use_mistral
# Import the SFT wrapper
from healthver_sft import verify_claim as use_healthver_sft

# Map model names to corresponding functions
MODEL_MAP = {
    "biogpt": use_biogpt,
    "llama2": use_llama2,
    "mistral": use_mistral,
    "healthver": use_healthver_sft,
}


def generate_verdict(claim: str, evidence: list[str], model: str) -> dict:
    """
    Run the selected model on the given claim and evidence.

    Args:
        claim: The medical claim to verify.
        evidence: A list of evidence strings.
        model: One of the keys in MODEL_MAP.

    Returns:
        A dict with keys 'verdict' and 'explanation'.
    """
    if model not in MODEL_MAP:
        raise ValueError(f"Unknown model '{model}'. Available: {list(MODEL_MAP.keys())}")
    return MODEL_MAP[model](claim, evidence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG LLM verifier")
    parser.add_argument("--claim", type=str, required=True, help="The claim to verify.")
    parser.add_argument(
        "--evidence", type=str, nargs='+', required=True,
        help="Evidence passages (as space-separated strings).")
    parser.add_argument(
        "--model", type=str, default="biogpt",
        choices=list(MODEL_MAP.keys()),
        help="Which LLM backend to use.")
    args = parser.parse_args()

    result = generate_verdict(args.claim, args.evidence, args.model)
    print("Verdict:", result.get("verdict"))
    print("Explanation:\n", result.get("explanation"))
