# make_healthver_sft.py
# Converts HealthVer into prompt/completion JSONL format for SFT

from datasets import load_dataset
import json, textwrap, pathlib

for split in ("train", "validation"):
    ds = load_dataset("dwadden/healthver_entailment", split=split)
    out_file = pathlib.Path(f"healthver_{split}.jsonl")

    with out_file.open("w") as f:
        for ex in ds:
            abstract = " ".join(ex["abstract"])
            claim = ex["claim"]
            verdict = ex["verdict"].upper().replace("NEUTRAL", "NOT ENOUGH INFO")
            justification = ex["abstract"][0][:180]  # crude summary from first sentence

            prompt = textwrap.dedent(f"""\
                You are a medical claim‑verification assistant.
                PubMed abstract: {abstract}

                Task: Return JSON {{verdict, justification}}.
                Verdict ∈ [SUPPORTED, REFUTED, NOT ENOUGH INFO].

                Claim: {claim}
                JSON:
            """).strip()

            completion = json.dumps({
                "verdict": verdict,
                "justification": justification + "…"
            })

            f.write(json.dumps({
                "prompt": prompt,
                "completion": completion
            }) + "\n")

    print(f"Wrote {len(ds)} examples to {out_file}")
