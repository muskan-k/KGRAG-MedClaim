FROM python:3.9-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --upgrade pip && \
    pip3 install spacy==3.8.5 scispacy && \
    pip3 uninstall -y numpy && \
    pip3 install numpy==1.26.4 && \
    pip3 install biopython networkx rdflib requests && \
    pip3 install matplotlib


COPY en_core_sci_lg.tar.gz .
RUN pip3 install en_core_sci_lg.tar.gz && \
    python3 -m spacy link en_core_sci_lg en_core_sci_lg

# Set working directory
WORKDIR /app

# Copy your script
COPY main.py .

# Run script
CMD ["python3", "main.py"]