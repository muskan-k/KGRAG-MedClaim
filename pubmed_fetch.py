import os
import re
import time
import json
from typing import List, Dict

import requests
from Bio import Entrez  # already installed in the base image
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
Entrez.email = os.getenv("ENTREZ_EMAIL", "rag@example.com")
Entrez.tool = "rag_pipeline_demo"

MAX_CANDIDATES = 100
DEFAULT_TOP_K = 5
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# ---------------------------------------------------------------------------
# BASIC TEXT NORMALISATION
# ---------------------------------------------------------------------------
_EN_STOP = {
    "a", "an", "the", "of", "to", "and", "in", "on", "for", "with", "into", "from", "by",
    "about", "is", "are", "was", "were", "be", "been", "being", "as", "that", "this",
    "these", "those", "it", "its", "their", "at", "or", "not", "but", "than", "then",
}

def _tokenise(text: str) -> List[str]:
    words = re.findall(r"[A-Za-z0-9-]+", text.lower())
    return [w for w in words if w not in _EN_STOP and len(w) > 2]

# ---------------------------------------------------------------------------
# MAIN QUERY + FETCH
# ---------------------------------------------------------------------------

def _build_pubmed_query(claim: str) -> str:
    # Use raw claim string as query (no MeSH, no TIAB)
    return claim.strip()


def _search_pubmed(term: str, retmax: int = MAX_CANDIDATES) -> List[str]:
    url = f"{NCBI_BASE}/esearch.fcgi"
    params = {"db": "pubmed", "term": term, "retmode": "json", "retmax": retmax, "api_key": os.getenv("ENTREZ_API_KEY", "")}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()["esearchresult"].get("idlist", [])


def _fetch_abstracts(pmids: List[str]) -> Dict[str, str]:
    if not pmids:
        return {}
    url = f"{NCBI_BASE}/efetch.fcgi"
    params = {
        "db": "pubmed", "id": ",".join(pmids), "retmode": "xml", "rettype": "abstract"
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    import xml.etree.ElementTree as ET

    root = ET.fromstring(r.content)
    out: Dict[str, str] = {}
    for art in root.findall(".//PubmedArticle"):
        pmid_el = art.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else None
        title_el = art.find(".//ArticleTitle")
        abst_el = art.find(".//Abstract/AbstractText")
        if pmid:
            title = title_el.text if title_el is not None else ""
            abst = abst_el.text if abst_el is not None else ""
            out[pmid] = f"{title}. {abst}"
    return out

# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------

def get_relevant_abstracts(claim: str, top_k: int = DEFAULT_TOP_K) -> List[str]:
    query = _build_pubmed_query(claim)
    print(f"[PubMed] query → {query}")

    pmids = _search_pubmed(query)
    if not pmids:
        return []

    abs_map = _fetch_abstracts(pmids)
    if not abs_map:
        return []

    corpus = list(abs_map.values())
    bm25 = BM25Okapi([_tokenise(d) for d in corpus])
    scores = bm25.get_scores(_tokenise(claim))

    ranked = sorted(zip(abs_map.keys(), corpus, scores), key=lambda x: x[2], reverse=True)
    return [f"{i+1}. {r[1]}" for i, r in enumerate(ranked[:top_k])]

# ---------------------------------------------------------------------------
# CLI test helper ------------------------------------------------------------
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Aspirin treats headache"
    for abs_txt in get_relevant_abstracts(q, 3):
        print("-" * 80)
        print(abs_txt[:1000])
