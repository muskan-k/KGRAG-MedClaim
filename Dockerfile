# ---------------- Dockerfile ----------------
FROM python:3.9-slim
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
        "rank-bm25==0.2.2" \
        datasets

# --- 3) install **local** model tarball --------------------------------------
COPY en_core_sci_lg.tar.gz /tmp/en_core_sci_lg-0.5.3.tar.gz
RUN pip install /tmp/en_core_sci_lg-0.5.3.tar.gz && \
    python -m spacy link en_core_sci_lg en_core_sci_lg && \
    rm /tmp/en_core_sci_lg-0.5.3.tar.gz

# --- 4) your application code ------------------------------------------------
COPY llama2.py query_kg.py .
COPY rag_llm.py pubmed_fetch.py rag_pipeline.py query_kg_utils.py /app/
COPY healthver_sft.py train_healthver_sft.py /app/

CMD ["python3", "llama2.py"]
