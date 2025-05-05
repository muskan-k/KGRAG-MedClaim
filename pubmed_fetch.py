import os
import re
import time
import json
from functools import lru_cache
from typing import List, Dict, Tuple

import requests
from Bio import Entrez  # already installed in the base image
from rank_bm25 import BM25Okapi  # tiny, pure‑python BM25 implementation

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
Entrez.email = os.getenv("ENTREZ_EMAIL", "mkothari@umass.edu")
Entrez.tool = "rag_pipeline_demo"

MAX_CANDIDATES = 100      # abstracts pulled down before local re‑rank
DEFAULT_TOP_K   = 5       # what we return to the pipeline
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
    """Lower‑case, strip punctuation, remove trivial stop‑words."""
    words = re.findall(r"[A-Za-z0-9-]+", text.lower())
    return [w for w in words if w not in _EN_STOP and len(w) > 2]

# ---------------------------------------------------------------------------
# GENERIC MESH EXPANSION (no hard‑coded dict!)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1024)
def _mesh_synonyms(term: str) -> List[str]:
    """
    Return MeSH headings PubMed’s automatic term mapping finds for *term*.
    Falls back to [] (meaning: no useful MeSH synonym).
    """
    try:
        url = f"{NCBI_BASE}/esearch.fcgi"
        resp = requests.get(
            url,
            params={"db": "pubmed", "term": term, "retmode": "json", "retmax": 0},
            timeout=6,
        )
        resp.raise_for_status()
        stack = resp.json()["esearchresult"].get("translationstack", [])

        mesh_terms: set[str] = set()

        def _collect(entry):
            """Pull “[MeSH Terms]” tokens out of dicts or strings."""
            if isinstance(entry, dict) and (entry.get("Field") or entry.get("field")) == "MeSH Terms":
                txt = entry["term"]
            elif isinstance(entry, str) and entry.endswith("[MeSH Terms]"):
                txt = entry
            else:
                return
            cleaned = re.sub(r"\[.*?]$", "", txt).strip('"').lower()
            if cleaned:
                mesh_terms.add(cleaned)

        for elem in stack:
            if isinstance(elem, list):
                for sub in elem:
                    _collect(sub)
            else:
                _collect(elem)

        # if the only thing we found *is* the original term, treat as “no synonym”
        if mesh_terms == {term.lower()}:
            return []
        return sorted(mesh_terms)

    except Exception:
        return []



# ---------------------------------------------------------------------------
# MAIN QUERY + FETCH
# ---------------------------------------------------------------------------

def _build_pubmed_query(claim: str, *, verbose: bool = False) -> str:
    toks = _tokenise(claim)
    mesh_chunks: list[str] = []

    if verbose:
        print("\n[MeSH groups]")

    for t in toks:
        syns = _mesh_synonyms(t)
        if syns:
            if verbose:
                print(f"{t:>12}  →  {', '.join(syns)}")
            mesh_chunks.extend([f'"{s}"[MeSH Terms]' for s in syns])

    # If *no* word produced a MeSH descriptor, fall back to the raw content words.
    if not mesh_chunks:
        mesh_chunks = [f"{w}[TIAB]" for w in toks]

    return " OR ".join(mesh_chunks)



def _search_pubmed(term: str, retmax: int = MAX_CANDIDATES) -> List[str]:
    """Return PubMed IDs matching *term* (already formatted)."""
    url = f"{NCBI_BASE}/esearch.fcgi"
    params = {"db": "pubmed", "term": term, "retmode": "json", "retmax": retmax}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()["esearchresult"].get("idlist", [])


def _fetch_abstracts(pmids: List[str]) -> Dict[str, str]:
    """Fetch title + abstract for every PMID given."""
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
    """Return *top_k* PubMed abstracts most relevant to *claim*.
    Uses generic MeSH expansion + local BM25 re‑rank so we never send the whole
    sentence to NCBI.
    """
    query = _build_pubmed_query(claim, verbose=True)
    print(f"[PubMed] query → {query}")  # keeps the CLI demo transparent

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
