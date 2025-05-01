import os
import json
import re
import requests
from urllib.parse import quote
from Bio import Entrez
import networkx as nx
import matplotlib.pyplot as plt
from rdflib import Graph as RDFGraph, Namespace, URIRef

# ─── 0) Configuration ─────────────────────────────────────────────────────────
Entrez.email = "mkothari@umass.edu"
MAX_WORDS = 5  # Limit to reduce verbosity

PROMPT_INSTRUCTIONS = """
You are an information extraction model for biomedical research abstracts.
You are a biomedical triplet extraction model.

Extract *complete* triplets of the form:
  - object: biomedical noun phrase
  - subject: biomedical noun phrase
  - predicate: short relation verb or verb phrase 

Respond with ONLY valid JSON like this:

{
  "triplets": [
    {"object": "headache", "subject": "aspirin", "predicate": "treats"},
    {"object": "COVID-19", "subject": "vaccine", "predicate": "prevents"}
  ]
}

Rules:
- Each triplet MUST contain all 3 fields: object, subject, predicate.
- Do not return any nulls or missing fields.
- Use clear biomedical nouns or noun phrases for object.
- Do not include explanation, markdown, or natural language text.
"""

def truncate_clean(text, max_words=3):
    bad_endings = {"in", "of", "for", "with", "and", "to", "against", "the"}
    words = text.strip().split()
    clean = " ".join(words[:max_words])
    while clean.split() and clean.split()[-1].lower() in bad_endings:
        clean = " ".join(clean.split()[:-1])
    return clean

def fetch_pubmed_abstracts(query, retmax=3):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax)
    record = Entrez.read(handle)
    ids = record["IdList"]
    if not ids:
        return []
    handle = Entrez.efetch(db="pubmed", id=",".join(ids), retmode="xml")
    docs = Entrez.read(handle)
    out = []
    for art in docs["PubmedArticle"]:
        pmid = art["MedlineCitation"]["PMID"]
        artinfo = art["MedlineCitation"]["Article"]
        abst = artinfo.get("Abstract", {}).get("AbstractText", [])
        abstract = " ".join(str(x) for x in abst) if isinstance(abst, list) else str(abst)
        out.append({"pmid": pmid, "abstract": abstract})
    return out

def extract_triples_llm(abstract_text):
    full_prompt = PROMPT_INSTRUCTIONS + "\n\nAbstract:\n" + abstract_text
    try:
        response = requests.post(
        "http://host.docker.internal:11434/api/generate",
        json={"model": "mistral", "prompt": prompt, "stream": False}
    )
        response.raise_for_status()
        content = response.json()["response"]

        match = re.search(r'\{.*"triplets"\s*:\s*\[.*?\]\s*\}', content, re.DOTALL)
        if not match:
            raise ValueError("No valid triplets JSON block found.")
        data = json.loads(match.group(0))

        triplets = []
        for t in data.get("triplets", []):
            s, p, o = t.get("subject"), t.get("predicate"), t.get("object")
            if s and p and o:
                triplets.append((truncate_clean(s), truncate_clean(p, 3), truncate_clean(o, 3)))
            else:
                print(f"⚠️ Malformed triplet skipped: {t}")
        return triplets
    except Exception as e:
        print(f"LLM extraction failed: {e}")
        return []

def build_nx_graph(triples):
    G = nx.DiGraph()
    for s, p, o in triples:
        G.add_node(s)
        G.add_node(o)
        G.add_edge(s, o, relation=p)
    return G

def visualize_graph(G, out_path="graph.png"):
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.5)
    nx.draw(G, pos, with_labels=True, node_color="skyblue", edge_color="gray", node_size=800, font_size=9)
    edge_labels = nx.get_edge_attributes(G, "relation")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Biomedical Knowledge Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"📸 Graph image saved: {out_path}")

def serialize_graphml(G, path="pubmed_kg_llm.graphml"):
    nx.write_graphml(G, path)

def build_rdf_graph(triples, namespace="http://example.org/"):
    g = RDFGraph()
    ns = Namespace(namespace)
    for s, p, o in triples:
        subj = URIRef(ns[quote(s.replace(" ", "_"), safe="")])
        pred = URIRef(ns[quote(p.replace(" ", "_"), safe="")])
        obj  = URIRef(ns[quote(o.replace(" ", "_"), safe="")])
        g.add((subj, pred, obj))
    return g

def main():
    print("Fetching PubMed abstracts...")
    articles = fetch_pubmed_abstracts("COVID-19 vaccine efficacy")

    print("Extracting triples with Mistral (Ollama)...")
    all_triples = []
    for art in articles:
        triples = extract_triples_llm(art["abstract"])
        all_triples.extend(triples)

    print("\nExtracted Triplets:")
    for t in triples:
        print(f"  - ({t[0]} → {t[1]} → {t[2]})")

    print("Building knowledge graph...")
    G = build_nx_graph(all_triples)
    print("\nKnowledge Graph (plain triples):")
    for u, v, data in G.edges(data=True):
        relation = data.get("relation", "related_to")
        print(f"{u} --[{relation}]--> {v}")

    print("Visualizing graph...")
    visualize_graph(G)

    print("Saving GraphML...")
    serialize_graphml(G)
    print(f"GraphML saved: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("Saving RDF...")
    rdf = build_rdf_graph(all_triples)
    rdf.serialize(destination="pubmed_kg_llm.ttl", format="turtle")
    print("RDF Turtle saved: pubmed_kg_llm.ttl")

if __name__ == "__main__":
    main()