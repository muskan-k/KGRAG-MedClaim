###############################################################################
# ─────────  Stage‑1  (GPU trainer / model builder)  ──────────────────────────
###############################################################################
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime AS trainer

# ---- tooling ---------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential && \
    rm -rf /var/lib/apt/lists/*

ENV TRANSFORMERS_NO_TORCHVISION=1

    # Force remove torchvision if auto-installed
RUN pip uninstall -y torchvision || true
    
    # Now install what you actually need
RUN pip install --no-cache-dir \
    "transformers==4.40.0" \
    "datasets" \
    "trl==0.8.6" \
    "bitsandbytes==0.43.0" \
    "sentencepiece" \
    "unsloth==2025.4.7"




# ---- bring in data + training script (kept outside runtime image) ----------
WORKDIR /train
#   copy whatever script you use for SFT
COPY scripts/train_healthver_sft.py ./
COPY scripts/make_healthver_sft.py  ./

# ---- run when you `--target trainer` ---------------------------------------
# CMD ["python", "train_healthver_sft.py"]
# (We leave CMD blank so the runtime stage becomes the default target.)

###############################################################################
# ─────────  Stage‑2  (slim runtime; keeps “vanilla” behaviour)  ─────────────
###############################################################################
FROM python:3.9-slim AS runtime 
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_INPUT=1
WORKDIR /app

# --- OS toolchain ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- 1) pin binary‑compatible NumPy first ------------------------------------
RUN pip install --upgrade pip && \
    pip install --no-cache-dir "numpy==1.24.4"

# --- 2) core project + NLP stack (spaCy 3.6 / sci‑spaCy 0.5.3) ---------------
RUN pip install --no-cache-dir \
        "biopython==1.84" \
        networkx rdflib requests matplotlib \
        "torch==2.1.*" \
        "transformers==4.38" \
        protobuf sacremoses "neo4j==5.19" accelerate tqdm \
        "spacy==3.6.1" \
        "scispacy==0.5.3" \
        "rank-bm25==0.2.2"

# --- 3) install **local** model tarball --------------------------------------
COPY en_core_sci_lg.tar.gz /tmp/en_core_sci_lg-0.5.3.tar.gz
RUN pip install /tmp/en_core_sci_lg-0.5.3.tar.gz && \
    python -m spacy link en_core_sci_lg en_core_sci_lg && \
    rm /tmp/en_core_sci_lg-0.5.3.tar.gz

# --- 4) your application code ------------------------------------------------
COPY llama2.py query_kg.py .
COPY rag_llm.py pubmed_fetch.py rag_pipeline.py query_kg_utils.py /app/

CMD ["python3", "llama2.py"]
