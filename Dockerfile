FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install all required libraries directly
RUN apt-get update && apt-get install -y build-essential curl && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --upgrade pip && \
    pip3 install \
        biopython>=1.84 \
        networkx \
        rdflib \
        requests \
        matplotlib \
        torch>=2.1 \
        transformers>=4.38 \
        protobuf \
        sacremoses \
        neo4j>=5.19 \
        accelerate \
        tqdm

# Copy Python script into image
COPY llama2.py .
COPY query_kg.py .

# ----- Vanilla RAG additions -----
COPY rag_llm.py pubmed_fetch.py rag_pipeline.py query_kg_utils.py /app/

# Run the script
CMD ["python3", "llama2.py"]