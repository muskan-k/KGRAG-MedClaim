###############################################################################
# ─────────  Stage-1  (GPU trainer / model builder)  ──────────────────────────
###############################################################################
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime AS trainer

# --- system tooling ----------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Avoid pulling in torchvision
ENV TRANSFORMERS_NO_TORCHVISION=1
RUN pip uninstall -y torchvision || true

# --- training libs ----------------------------------------------------------
RUN pip install --no-cache-dir \
    "transformers>=4.46.1,<4.47.0" \
    "datasets" \
    "accelerate"

# --- copy training scripts -------------------------------------------------
WORKDIR /train
COPY scripts/train_healthver_sft.py ./
COPY scripts/make_healthver_sft.py  ./
# Note: mount your healthver_train.jsonl & healthver_validation.jsonl at /train when you run

###############################################################################
# ─────────  Stage-2  (slim runtime; inference & KG-RAG)  ─────────────────────
###############################################################################
FROM python:3.9-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_INPUT=1

WORKDIR /app

# --- OS toolchain ------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# --- pin NumPy ---------------------------------------------------------------
RUN pip install --upgrade pip && \
    pip install --no-cache-dir "numpy==1.24.4"

# --- inference-time libs ----------------------------------------------------
RUN pip install --no-cache-dir \
    biopython==1.84 \
    networkx rdflib requests matplotlib \
    torch==2.1.* \
    "transformers>=4.46.1,<4.47.0" \
    spacy==3.6.1 \
    scispacy==0.5.3 \
    rank-bm25==0.2.2

# --- install local sci-spaCy model ------------------------------------------
COPY en_core_sci_lg.tar.gz /tmp/
RUN pip install /tmp/en_core_sci_lg-0.5.3.tar.gz && \
    python -m spacy link en_core_sci_lg en_core_sci_lg && \
    rm /tmp/en_core_sci_lg-0.5.3.tar.gz

# --- copy your inference pipeline code --------------------------------------
COPY llama2.py query_kg.py /app/
COPY rag_llm.py pubmed_fetch.py rag_pipeline.py query_kg_utils.py /app/

CMD ["python3", "llama2.py"]
