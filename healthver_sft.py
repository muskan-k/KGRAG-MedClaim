import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# 1. LOAD your fine-tuned checkpoint
MODEL_DIR = "healthver-sft"
SEQ_LEN = 400

# 2. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
tokenizer.model_max_length = SEQ_LEN

# 3. Model
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
# Move model to GPU and use mixed precision
if torch.cuda.is_available():
    model = model.half().to("cuda")
else:
    model = model.to("cpu")

# 4. Generator pipeline with device assignment
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    truncation=True,
    device=0 if torch.cuda.is_available() else -1,
)


def verify_claim(claim: str, evidence: list[str]) -> dict:
    """
    Same signature as biogpt/llama2/mistral.
    Returns {"verdict": "...", "explanation": "..."}.
    """
    # build the prompt
    prompt = (
            "Evidence:\n" + "\n".join(evidence) +
            f"\n\nClaim: {claim}\n"
            "Question: Based on the above, is this claim SUPPORTED, CONTRADICTED, or UNCERTAIN? "
            "Please give a one-sentence verdict followed by a brief justification."
    )
    out = generator(prompt)[0]["generated_text"]
    # naive parse: assumes model writes "VERDICT: ...\nEXPLANATION: ..."
    lines = out.splitlines()
    verdict = None
    explanation = []
    for line in lines:
        if line.upper().startswith(("VERDICT:", "SUPPORT", "CONTRADICTION", "UNCERTAIN")):
            # capture after the colon or full line if no colon
            verdict = line.split(":", 1)[-1].strip() if ":" in line else line.strip()
        else:
            explanation.append(line.strip())
    return {
        "verdict": verdict or lines[0].strip(),
        "explanation": " ".join(explanation).strip(),
    }