"""
query_kg_utils.py - Light Neo4j helper used by rag_pipeline.py.
Assumes Neo4j container credentials via env vars.
"""
import os
from neo4j import GraphDatabase
from functools import lru_cache
from typing import List, Tuple

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
        
    #     query = """
    #     MATCH (s)-[r]->(o)
    #     WHERE toLower(s.name) CONTAINS toLower($claim)
    #        OR toLower(o.name) CONTAINS toLower($claim)
    #     RETURN s.name AS subj, type(r) AS rel, o.name AS obj
    #     LIMIT $limit
    #     """
    #     with self.driver.session() as session:
    #         result = session.run(query, claim=claim, limit=limit)
    #         return [(rec["subj"], rec["rel"], rec["obj"]) for rec in result]
    
    def get_triples(self, claim: str, limit: int = 20) -> List[Tuple[str, str, str]]:
        claim = [w.lower() for w in claim.split()]
        query = """
        MATCH (s)-[r]->(o)
        WHERE ANY(w IN $claim WHERE
                toLower(s.name) CONTAINS w OR
                toLower(o.name) CONTAINS w)
        RETURN s.name AS s, type(r) AS r, o.name AS o
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, claim=claim, limit=limit)
            return [(rec["s"], rec["r"], rec["o"]) for rec in result]

_client = KGClient()

@lru_cache(maxsize=128)
def get_triples_for_claim(claim: str, limit: int = 20):
    return _client.get_triples(claim, limit)
