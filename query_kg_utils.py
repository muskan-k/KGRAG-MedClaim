"""
query_kg_utils.py - Light Neo4j helper used by rag_pipeline.py.
Assumes Neo4j container credentials via env vars.
"""
import os
from neo4j import GraphDatabase
from functools import lru_cache
from typing import List, Tuple
import scispacy
from neo4j import GraphDatabase
import spacy
_nlp = spacy.load("en_core_sci_lg")     # loads at container start‑up

STOP_POS = {"DET", "ADP", "CCONJ"}      # skip articles, preps, etc.

def _extract_entities(text: str)  -> List[str]:
    doc = _nlp(text)
    ents = {ent.text.lower() for ent in doc.ents}
    # Fallback: also grab bare nouns if NER finds nothing
    if not ents:
        ents = {t.lemma_.lower() for t in doc if t.pos_ not in STOP_POS and len(t) > 2}
    return list(ents)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

class KGClient:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    # def get_triples(self, claim: str, limit: int = 20) -> List[Tuple[str, str, str]]:
    #     claim = [w.lower() for w in claim.split()]
    #     query = """
    #     MATCH (s)-[r]->(o)
    #     WHERE ANY(w IN $claim WHERE
    #             toLower(s.name) CONTAINS w OR
    #             toLower(o.name) CONTAINS w)
    #     RETURN s.name AS s, type(r) AS r, o.name AS o
    #     LIMIT $limit
    #     """
    #     with self.driver.session() as session:
    #         result = session.run(query, claim=claim, limit=limit)
    #         return [(rec["s"], rec["r"], rec["o"]) for rec in result]
    

    def get_triples(self, claim: str, limit: int = 20):
        """
        Return up to `limit` triples whose subject *or* object name is “similar”
        to any entity extracted from the claim sentence.

        Similarity rule here =   a CONTAINS b   OR   b CONTAINS a
        (i.e. substring / prefix match, case‑insensitive).
        """
        ents = _extract_entities(claim)
        if not ents:
            return []

        ents_low = [e.lower() for e in ents]          # normalise once

        cypher = """
        UNWIND $ents AS term                          // iterate over query tokens
        MATCH (s)-[r]->(o)
        WHERE toLower(s.name)  CONTAINS term          // subject ≈ term
        OR term CONTAINS toLower(s.name)           // term ≈ subject
        OR toLower(o.name)  CONTAINS term          // object ≈ term
        OR term CONTAINS toLower(o.name)           // term ≈ object
        RETURN DISTINCT s.name  AS subj,
                        type(r) AS rel,
                        o.name  AS obj
        LIMIT $lim
        """

        with self.driver.session() as sess:
            res = sess.run(cypher, ents=ents_low, lim=limit)
            return [(row["subj"], row["rel"], row["obj"]) for row in res]


_client = KGClient()

@lru_cache(maxsize=128)
def get_triples_for_claim(claim: str, limit: int = 20):
    return _client.get_triples(claim, limit)
