FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install all required libraries directly
RUN apt-get update && apt-get install -y build-essential curl && \
    rm -rf /var/lib/apt/lists/* && \
    pip3 install --upgrade pip && \
    # pip3 uninstall -y numpy && \
    # pip3 install numpy==1.26.4 && \
    pip3 install biopython networkx rdflib requests matplotlib torch transformers protobuf sacremoses neo4j

# Copy Python script into image
COPY llama2.py .
COPY query_kg.py .

# Run the script
CMD ["python3", "llama2.py"]