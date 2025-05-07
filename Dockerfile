# ─────────────────────────────────────────────────────────────────────────────
# BUILDER: install Python deps once
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# 1) System libs
RUN apt-get update \
 && apt-get install -y --no-install-recommends git build-essential \
 && rm -rf /var/lib/apt/lists/*

# 2) Pin numpy early
RUN pip install --upgrade pip \
 && pip install numpy==1.24.4

# 3) Copy and install Python requirements
COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt
RUN pip uninstall -y bitsandbytes || true
# also keep this env so Transformers never even tries
ENV TRANSFORMERS_NO_BITSANDBYTES=1
# ─────────────────────────────────────────────────────────────────────────────
# RUNTIME: brings in deps + your code + data
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime
WORKDIR /app

# 1) Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# 2) Copy your application code and data
COPY scripts/ ./scripts/          
COPY data/      ./data/          
COPY models/    ./models/             
COPY scripts/train_sft.py           ./train_sft.py
COPY scripts/finetuned.py           ./finetuned.py
COPY scripts/evaluate_healthver.py  ./evaluate_healthver.py
COPY scripts/merge_adapter.sh ./scripts/merge_adapter.sh

ENTRYPOINT ["bash", "-lc", "\
  set -euo pipefail; \
  echo '=== (1) TRAIN SFT ==='; \
  python scripts/train_sft.py; \
  echo '=== (2) MERGE ADAPTER ==='; \
  bash scripts/merge_adapter.sh; \
  echo '=== (3) EVALUATE ==='; \
  python scripts/evaluate_healthver.py --model_dir=models/llama2-healthver-sft-merged --data_dir=data/healthver --max_samples=10 \
"]