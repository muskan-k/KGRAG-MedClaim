import os
from neo4j import GraphDatabase

def fetch_triplets_from_neo4j():
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "test123")

    driver = GraphDatabase.driver(uri, auth=(user, password))

    query = """
    MATCH (s:Entity)-[r:RELATION]->(o:Entity)
    RETURN s.name AS subject, r.type AS predicate, o.name AS object
    """

    with driver.session() as session:
        results = session.run(query)
        triplets = [(record["subject"], record["predicate"], record["object"]) for record in results]

    driver.close()
    return triplets

def main():
    print("🔁 Retrieving triplets from Neo4j...")
    triplets = fetch_triplets_from_neo4j()
    for s, p, o in triplets:
        print(f"  - ({s} → {p} → {o})")

if __name__ == "__main__":
    main()