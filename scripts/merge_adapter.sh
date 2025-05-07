#!/usr/bin/env bash
set -euo pipefail

python - <<'PYCODE'
from finetuned import load
model, tok = load("distilgpt2::models/distilgpt2-healthver-sft")
model.save_pretrained("models/distilgpt2-healthver-sft-merged")
tok.save_pretrained("models/distilgpt2-healthver-sft-merged")
PYCODE
