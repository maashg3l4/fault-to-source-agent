import networkx as nx
from neo4j import GraphDatabase
import os

URI = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

def import_gml_to_neo4j(gml_path):
    G = nx.read_gml(gml_path)
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    
    with driver.session() as session:
        # Clear existing graph (optional, for a clean start)
        session.run("MATCH (n) DETACH DELETE n")
        
        # Create nodes
        for node_id, attrs in G.nodes(data=True):
            label = attrs.get('label', 'Variable')
            session.run(
                f"CREATE (n:{label} {{id: $id, name: $name}})",
                id=node_id, name=attrs.get('name', node_id)
            )
        
        # Create edges
        for source, target, attrs in G.edges(data=True):
            session.run(
                """
                MATCH (a {id: $source}), (b {id: $target})
                CREATE (a)-[:CAUSES]->(b)
                """,
                source=source, target=target
            )
    
    driver.close()
    print(f"✅ Imported {G.number_of_nodes()} nodes and {G.number_of_edges()} edges into Neo4j.")

if __name__ == "__main__":
    # Adjust the path based on your extraction
    import_gml_to_neo4j("data/raw/causRCA/expert_graph/expert_graph.gml")