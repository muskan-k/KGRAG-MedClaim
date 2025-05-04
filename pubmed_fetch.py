"""
pubmed_fetch.py - Minimal PubMed fetcher using NCBI E‑utilities via Biopython.
Usage: abstracts = fetch_abstracts("metformin insulin sensitivity", max_hits=20)
"""
from typing import List
from Bio import Entrez
import os
import re

Entrez.email = os.getenv("ENTREZ_EMAIL", "mkothari@umass.edu")

def fetch_abstracts(query: str, max_hits: int = 20) -> List[str]:
    search_handle = Entrez.esearch(db="pubmed", term=query, retmax=max_hits)
    search_results = Entrez.read(search_handle)
    pmids = search_results.get("IdList", [])
    if not pmids:
        return []
    fetch_handle = Entrez.efetch(
        db="pubmed",
        id=",".join(pmids),
        rettype="abstract",
        retmode="text",
    )
    raw_text = fetch_handle.read()
    abstracts = [sec.strip() for sec in re.split(r"\n\s*\n", raw_text) if sec.strip()]
    return abstracts
