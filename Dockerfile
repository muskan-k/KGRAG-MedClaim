###############################################################################
# ─────────  Stage-1  (GPU trainer / model builder)  ──────────────────────────
###############################################################################
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime AS trainer

# keep Python lean
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_INPUT=1

# OS-level tooling for building & Git
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies for full-parameter SFT on CPU
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
      "numpy<2" \
      torch==2.1.* \
      "transformers>=4.46.1,<4.47.0" \
      datasets \
      accelerate \
      trl \
      peft \
      bitsandbytes

# copy in your SFT scripts (JSONLs will be mounted at runtime)
WORKDIR /train
COPY scripts/train_healthver_sft.py ./
COPY scripts/make_healthver_sft.py ./

# no CMD here: you’ll override with `docker run … python3 train_healthver_sft.py`

###############################################################################
# ─────────  Stage-2  (slim runtime; inference & KG-RAG)  ─────────────────────
###############################################################################
FROM python:3.9-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_INPUT=1

# OS toolchain
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential && \
    rm -rf /var/lib/apt/lists/*

# pin NumPy, then inference-only Python libs
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
      numpy==1.24.4 \
      torch==2.1.* \
      "transformers>=4.46.1,<4.47.0" \
      biopython==1.84 \
      networkx rdflib requests matplotlib \
      spacy==3.6.1 \
      scispacy==0.5.3 \
      rank-bm25==0.2.2

# install your local sci-spaCy model
COPY en_core_sci_lg.tar.gz /tmp/
RUN pip install /tmp/en_core_sci_lg-0.5.3.tar.gz && \
    python -m spacy link en_core_sci_lg en_core_sci_lg && \
    rm /tmp/en_core_sci_lg-0.5.3.tar.gz

# copy your inference pipeline code
WORKDIR /app
COPY llama2.py query_kg.py rag_llm.py pubmed_fetch.py rag_pipeline.py query_kg_utils.py ./

CMD ["python3", "llama2.py"]
